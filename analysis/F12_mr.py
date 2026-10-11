"""Two-sample cis-pQTL Mendelian randomization: intestinal epithelial proteins -> digestive outcomes in FinnGen R12.

Exposure instruments: conditionally independent cis-pQTL signals from UKB-PPP (Sun et al., Nature 2023, Supplementary
Table 16; SuSiE top variants, European discovery cohort), genome-wide significant (-log10 P >= 7.3), PIP >= 0.5, and
mutually independent (r2 < 0.01 with the other selected signals, from the LD values reported in the same table).
ST16 reports effects per SD of the standardised genotype; they are converted to per-allele effects by dividing by
sqrt(2p(1-p)) (checked against the per-allele discovery estimates in Supplementary Table 9 for the lead variants).
Effects are per alternative allele (A1) in SD units of protein. Outcomes: FinnGen R12 (no UK Biobank overlap).
Estimators: Wald ratio (single SNP); inverse-variance weighted (fixed effects, or multiplicative random effects when
Cochran's Q indicates heterogeneity); MR-Egger and weighted median when >= 3 instruments.
"""
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from common import PROJ, RES

MR = PROJ / "data" / "mr"
PROTS = ["GUCA2A", "FABP6", "DDC", "FABP2", "CDH17"]
OUTCOMES = {"K11_CHOLELITH": "Cholelithiasis", "K11_IBD_STRICT": "Inflammatory bowel disease",
            "K11_CD_STRICT2": "Crohn's disease", "K11_UC_STRICT2": "Ulcerative colitis", "K11_ACUTPANC": "Acute pancreatitis"}
R2_MAX = 0.01


def instruments():
    s = pd.read_csv(MR / "ukbppp" / "ST16_selected_proteins.csv")
    s.columns = ["pid", "vid", "rsid", "pip", "logbf", "beta", "se", "mlogp", "altfreq", "cistrans", "region", "ld",
                 "cs_size", "cs_ids", "cs_rsids", "cs_pips"]
    s["protein"] = s.pid.str.split(":").str[0]
    s = s[(s.cistrans == "cis") & (s.mlogp >= 7.3) & (s.pip >= 0.5) & s.rsid.str.startswith("rs")]
    s[["chr", "pos19", "a0", "a1"]] = s.vid.str.split(":", expand=True).iloc[:, :4]
    scale = np.sqrt(2 * s.altfreq * (1 - s.altfreq))
    s["beta"], s["se"] = s.beta / scale, s.se / scale
    keep = []
    for p, g in s.sort_values("mlogp", ascending=False).groupby("protein", sort=False):
        chosen = []
        for _, r in g.iterrows():
            ld = dict((m.group(1), float(m.group(2))) for m in re.finditer(r"([\w:]+:imp:v1) \(([\d.eE-]+)\)", str(r.ld)))
            if all(ld.get(c.vid, 0) < R2_MAX for c in chosen):
                chosen.append(r)
        keep += chosen
    return pd.DataFrame(keep)


def outcome_rows(code, rsids):
    f = MR / "finngen_r12" / f"finngen_R12_{code}.gz"
    pat = MR / "rsids.txt"
    pat.write_text("\n".join(rsids) + "\n")
    cmd = f"gzip -dc '{f}' | awk 'NR==1' ; gzip -dc '{f}' | grep -wF -f '{pat}'"
    out = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True).stdout
    from io import StringIO
    return pd.read_csv(StringIO(out), sep="\t")


def harmonise(inst, o):
    o = o.rename(columns={"#chrom": "chrom"})
    rows = []
    for _, r in inst.iterrows():
        m = o[o.rsids.astype(str).str.split(",").apply(lambda xs: r.rsid in xs)]
        if m.empty:
            continue
        m = m.iloc[0]
        if m.alt == r.a1 and m.ref == r.a0:
            b, af = m.beta, m.af_alt
        elif m.alt == r.a0 and m.ref == r.a1:
            b, af = -m.beta, 1 - m.af_alt
        else:
            continue
        palin = {r.a0, r.a1} in ({"A", "T"}, {"C", "G"})
        if palin and 0.42 <= r.altfreq <= 0.58:
            continue
        rows.append({"protein": r.protein, "rsid": r.rsid, "ea": r.a1, "oa": r.a0, "eaf_exp": r.altfreq, "eaf_out": af,
                     "beta_exp": r.beta, "se_exp": r.se, "F": (r.beta / r.se) ** 2, "beta_out": b, "se_out": m.sebeta,
                     "p_out": m.pval})
    return pd.DataFrame(rows)


def mr(h):
    bx, by, sx, sy = h.beta_exp.to_numpy(), h.beta_out.to_numpy(), h.se_exp.to_numpy(), h.se_out.to_numpy()
    ratio, rse = by / bx, sy / np.abs(bx)
    out = []
    if len(h) == 1:
        return [{"method": "Wald ratio", "nsnp": 1, "b": ratio[0], "se": rse[0]}]
    w = 1 / rse ** 2
    b = np.sum(w * ratio) / np.sum(w)
    se = np.sqrt(1 / np.sum(w))
    q = np.sum(w * (ratio - b) ** 2)
    qp = stats.chi2.sf(q, len(h) - 1)
    phi = max(1, q / (len(h) - 1))
    out.append({"method": "IVW", "nsnp": len(h), "b": b, "se": se * np.sqrt(phi), "Q": q, "Q_p": qp})
    if len(h) >= 3:
        sign = np.sign(bx)
        X = np.column_stack([np.ones(len(h)), np.abs(bx)])
        W = np.diag(1 / sy ** 2)
        beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ (by * sign))
        resid = by * sign - X @ beta
        s2 = max(1, (resid ** 2 / sy ** 2).sum() / (len(h) - 2))
        cov = np.linalg.inv(X.T @ W @ X) * s2
        out.append({"method": "MR-Egger", "nsnp": len(h), "b": beta[1], "se": np.sqrt(cov[1, 1]),
                    "egger_intercept": beta[0], "egger_p": 2 * stats.t.sf(abs(beta[0] / np.sqrt(cov[0, 0])), len(h) - 2)})
        o = np.argsort(ratio)
        wr = (1 / rse[o] ** 2); cw = (np.cumsum(wr) - wr / 2) / wr.sum()
        wm = np.interp(0.5, cw, ratio[o])
        boots = []
        rng = np.random.default_rng(1)
        for _ in range(1000):
            rb = rng.normal(ratio, rse); ob = np.argsort(rb); wb = 1 / rse[ob] ** 2
            boots.append(np.interp(0.5, (np.cumsum(wb) - wb / 2) / wb.sum(), rb[ob]))
        out.append({"method": "Weighted median", "nsnp": len(h), "b": wm, "se": np.std(boots)})
    return out


if __name__ == "__main__":
    inst = instruments()
    inst = inst[inst.protein.isin(PROTS)]
    print(inst[["protein", "rsid", "a0", "a1", "altfreq", "beta", "se", "mlogp", "pip"]].to_string(index=False))
    rows, snps = [], []
    for code, label in OUTCOMES.items():
        f = MR / "finngen_r12" / f"finngen_R12_{code}.gz"
        if not f.exists():
            print("missing", f); continue
        o = outcome_rows(code, inst.rsid.tolist())
        h = harmonise(inst, o)
        h["outcome"] = label
        snps.append(h)
        for p, g in h.groupby("protein"):
            for r in mr(g):
                r.update(protein=p, outcome=label, OR=np.exp(r["b"]), lci=np.exp(r["b"] - 1.96 * r["se"]),
                         uci=np.exp(r["b"] + 1.96 * r["se"]), p=2 * stats.norm.sf(abs(r["b"] / r["se"])))
                rows.append(r)
    res = pd.DataFrame(rows)
    pd.concat(snps).to_csv(RES / "mr_harmonised_snps.csv", index=False)
    res.to_csv(RES / "mr_results.csv", index=False)
    show = res.assign(c=res.apply(lambda s: f"{s.OR:.2f} ({s.lci:.2f}-{s.uci:.2f}) p={s.p:.2g} [{int(s.nsnp)}]", axis=1))
    print(show.pivot_table(index=["protein", "method"], columns="outcome", values="c", aggfunc="first").to_string())
    if "Q" in res:
        print(res[res.method == "IVW"][["protein", "outcome", "Q", "Q_p"]].round(3).to_string(index=False))
