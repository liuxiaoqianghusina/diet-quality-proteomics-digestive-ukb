"""GUCA2A cis region: colocalisation (coloc.abf) with FinnGen R12 outcomes and MR with marginal cis-pQTL estimates.

Exposure: UKB-PPP discovery (European, n = 33,494) REGENIE summary statistics for GUCA2A (chr1, GENPOS in hg38,
BETA per ALLELE1 on the protein scale). Outcomes: FinnGen R12 (hg38; effect allele = alt).
Region: gene start +/- 1 Mb. Variants matched on hg38 position and alleles.
coloc.abf (Giambartolomei et al. 2014): prior variance W = 0.15^2 (protein, quantitative) and 0.2^2 (log OR);
priors p1 = p2 = 1e-4, p12 = 1e-5 (sensitivity 5e-6). Assumes at most one causal variant per trait in the region.
MR: Wald ratio for (a) the strongest common cis-pQTL (MAF >= 1%) and (b) the lead cis-pQTL overall; IVW over the
SuSiE-independent signals (Supplementary Table 16 of Sun et al.) using marginal per-allele estimates.
"""
import subprocess
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from common import PROJ, RES

MR = PROJ / "data" / "mr"
OUT = {"K11_CHOLELITH": "Cholelithiasis", "K11_IBD_STRICT": "Inflammatory bowel disease",
       "K11_CD_STRICT2": "Crohn's disease", "K11_UC_STRICT2": "Ulcerative colitis", "K11_ACUTPANC": "Acute pancreatitis"}
REGIONS = {"GUCA2A": (1, 42162690), "FABP6": (5, 160187367), "DDC": (7, 50458436), "FABP2": (4, 119317250),
           "CDH17": (8, 94127162)}
FLANK = 1_000_000


def finngen_regions():
    """One pass per FinnGen file; caches every protein's cis region (+/- 1 Mb) to data/mr/finngen_regions/."""
    d = MR / "finngen_regions"; d.mkdir(exist_ok=True)
    cond = " || ".join(f'($1=="{c}" && $2>={p - FLANK} && $2<={p + FLANK})' for c, p in REGIONS.values())
    for code in OUT:
        f = d / f"{code}.tsv"
        if f.exists():
            continue
        src = MR / "finngen_r12" / f"finngen_R12_{code}.gz"
        subprocess.run(["bash", "-c", f"gzip -dc '{src}' | awk -F'\\t' 'NR==1 || {cond}' > '{f}.tmp' && mv '{f}.tmp' '{f}'"],
                       check=True)
        print("cached", code, flush=True)


def logabf(beta, se, W):
    V = se ** 2
    r = W / (W + V)
    z = beta / se
    return 0.5 * (np.log(1 - r) + r * z ** 2)


def coloc_abf(l1, l2, p1=1e-4, p2=1e-4, p12=1e-5):
    """Posterior probabilities of H0-H4 (same algebra as coloc::coloc.abf, computed on the log scale)."""
    lsum = lambda x: np.max(x) + np.log(np.sum(np.exp(x - np.max(x))))
    a1, a2, a12 = lsum(l1), lsum(l2), lsum(l1 + l2)
    d = a12 - (a1 + a2)                                   # always < 0
    lH3 = np.log(p1) + np.log(p2) + a1 + a2 + np.log(-np.expm1(d))
    lH = np.array([0.0, np.log(p1) + a1, np.log(p2) + a2, lH3, np.log(p12) + a12])
    return np.exp(lH - lsum(lH))


if __name__ == "__main__":
    finngen_regions()
    chrom, start = REGIONS["GUCA2A"]
    f = next((MR / "ukbppp" / "GUCA2A_chr1").rglob("*.gz"))
    p = pd.read_csv(f, sep=" ", usecols=["GENPOS", "ID", "ALLELE0", "ALLELE1", "A1FREQ", "INFO", "N", "BETA", "SE", "LOG10P"])
    p = p[(p.GENPOS >= start - FLANK) & (p.GENPOS <= start + FLANK) & (p.INFO >= 0.8)]
    p["maf"] = np.minimum(p.A1FREQ, 1 - p.A1FREQ)
    print("UKB-PPP GUCA2A region variants:", len(p), "| lead:", p.loc[p.LOG10P.idxmax(), ["ID", "BETA", "LOG10P", "A1FREQ"]].tolist())
    st16 = pd.read_csv(MR / "ukbppp" / "ST16_selected_proteins.csv")
    st16 = st16[st16.iloc[:, 0].str.startswith("GUCA2A") & (st16.iloc[:, 9] == "cis") & (st16.iloc[:, 3] >= 0.5)
                & (st16.iloc[:, 7] >= 7.3)]
    indep_ids = st16.iloc[:, 1].tolist()

    rows_c, rows_m, snps = [], [], []
    for code, label in OUT.items():
        o = pd.read_csv(MR / "finngen_regions" / f"{code}.tsv", sep="\t")
        o = o[o["#chrom"].astype(str) == str(chrom)]
        m = p.merge(o, left_on="GENPOS", right_on="pos")
        same = (m.ALLELE1 == m.alt) & (m.ALLELE0 == m.ref)
        flip = (m.ALLELE1 == m.ref) & (m.ALLELE0 == m.alt)
        m = m[same | flip].copy()
        m["beta_out"] = np.where(same[same | flip], m.beta, -m.beta)
        m["se_out"] = m.sebeta
        # ---- coloc
        l1, l2 = logabf(m.BETA.to_numpy(), m.SE.to_numpy(), 0.15 ** 2), logabf(m.beta_out.to_numpy(), m.se_out.to_numpy(), 0.2 ** 2)
        for p12 in (1e-5, 5e-6):
            pp = coloc_abf(l1, l2, p12=p12)
            post = np.exp(l1 + l2 - (np.max(l1 + l2) + np.log(np.sum(np.exp(l1 + l2 - np.max(l1 + l2))))))
            top = m.iloc[int(np.argmax(post))]
            rows_c.append({"outcome": label, "p12": p12, "nsnps": len(m), **{f"PP.H{i}": pp[i] for i in range(5)},
                           "top_shared_variant": top.ID, "top_variant_rsid": top.rsids, "SNP.PP.H4": post.max(),
                           "outcome_lead_p_in_region": o.pval.min()})
        # ---- MR
        def wald(r, label_inst):
            b, se = r.beta_out / r.BETA, r.se_out / abs(r.BETA)
            return {"outcome": label, "instrument": label_inst, "variant": r.ID, "rsid": r.rsids, "eaf": r.A1FREQ,
                    "beta_exp": r.BETA, "F": (r.BETA / r.SE) ** 2, "beta_out": r.beta_out, "se_out": r.se_out,
                    "nsnp": 1, "b": b, "se": se}
        common = m[(m.maf >= 0.01) & (m.LOG10P >= 7.3)]
        if len(common):
            rows_m.append(wald(common.loc[common.LOG10P.idxmax()], "lead common cis-pQTL (MAF >= 1%)"))
        lead = m.loc[m.LOG10P.idxmax()]
        if lead.LOG10P >= 7.3:
            rows_m.append(wald(lead, "lead cis-pQTL overall"))
        ind = m[m.ID.isin(indep_ids)]
        if len(ind) >= 2:
            ratio, rse = ind.beta_out / ind.BETA, ind.se_out / ind.BETA.abs()
            w = 1 / rse ** 2
            b = (w * ratio).sum() / w.sum(); se = np.sqrt(1 / w.sum())
            q = (w * (ratio - b) ** 2).sum(); phi = max(1, q / (len(ind) - 1))
            rows_m.append({"outcome": label, "instrument": "IVW, SuSiE-independent signals (marginal estimates)",
                           "variant": ";".join(ind.ID), "rsid": ";".join(ind.rsids.astype(str)), "nsnp": len(ind),
                           "b": b, "se": se * np.sqrt(phi), "Q": q, "Q_p": stats.chi2.sf(q, len(ind) - 1)})
        snps.append(ind.assign(outcome=label))
    rc = pd.DataFrame(rows_c)
    rm = pd.DataFrame(rows_m)
    rm["OR"], rm["lci"], rm["uci"] = np.exp(rm.b), np.exp(rm.b - 1.96 * rm.se), np.exp(rm.b + 1.96 * rm.se)
    rm["p"] = 2 * stats.norm.sf((rm.b / rm.se).abs())
    rc.to_csv(RES / "coloc_guca2a.csv", index=False)
    rm.to_csv(RES / "mr_guca2a_full.csv", index=False)
    pd.concat(snps).to_csv(RES / "mr_guca2a_instruments.csv", index=False)
    pd.set_option("display.width", 250)
    print(rc.round(4).to_string(index=False))
    print(rm[["outcome", "instrument", "rsid", "nsnp", "OR", "lci", "uci", "p"]].round(4).to_string(index=False))
