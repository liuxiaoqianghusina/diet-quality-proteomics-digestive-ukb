"""Main figures (Arial throughout; italic P; SVG/PDF/PNG/TIFF at 600 dpi).
Fig 1 study design; Fig 2 diet-proteome volcano plots; Fig 3 signature-outcome forest plot (M1 vs M3);
Fig 4 intestinal epithelial proteins: food-group drivers and gallstone mediation.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from common import EXPOSURES, FIG, NEGATIVE, OUT_LABEL, POSITIVE, PRIMARY, RES

plt.rcParams.update({"font.family": "Arial", "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                     "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold", "font.size": 8,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "svg.fonttype": "none", "pdf.fonttype": 42})
SHORT = {"UPF": "UPF", "uPDI": "uPDI", "hPDI": "hPDI", "fibre": "Fibre"}
FG_LABEL = {"ssb": "Sugar-sweetened beverages", "sweets_desserts": "Sweets and desserts", "tea_coffee": "Tea and coffee",
            "misc_animal": "Miscellaneous animal foods", "fish_seafood": "Fish and seafood", "vegetable_oils": "Vegetable oils",
            "whole_grains": "Whole grains", "refined_grains": "Refined grains", "fruit_juices": "Fruit juices",
            "animal_fat": "Animal fat"}
EPI = ["guca2a", "ddc", "fabp2", "fabp6", "cdh17", "dsg2", "lgals4", "tff2"]
BLUE, RED, GREY = "#2b6cb0", "#c53030", "#c8c8c8"


def save(fig, name):
    for ext in ["svg", "pdf", "png", "tiff"]:
        fig.savefig(FIG / f"{name}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 1 design
def fig1():
    flow = pd.read_csv(RES / "flow_counts.csv", index_col=0)["n"]
    perf = pd.read_csv(RES / "signature_performance.csv").set_index("signature")
    sig_n = int(pd.read_csv(RES / "sample_sizes.csv", index_col=0).loc["independent signature sample", "n"])
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")

    def box(x, y, w, h, title, body, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=color, ec="#4a4a4a", lw=0.6))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=8, fontweight="bold")
        ax.text(x + w / 2, y + h - 6.5, body, ha="center", va="top", fontsize=6.3, linespacing=1.35)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#4a4a4a"))

    box(1, 30, 28, 18, "Dietary assessment",
        f"UK Biobank 24-h recalls\n(2009–2012), n = {flow['24h recall >=1 (whole cohort)']:,}\n"
        "UPF (item-level NOVA), hPDI,\nuPDI, energy-adjusted fibre", "#eef4fb")
    box(36, 30, 28, 18, "Diet–proteome sample",
        f"Randomly selected Olink\nparticipants with ≥1 valid recall\nn = {flow['... with >=1 valid 24h recall']:,}\n"
        f"2,919 plasma proteins\nlinear models + elastic net", "#fdf3e7")
    box(71, 30, 28, 18, "Proteomic signatures",
        f"Training 70% (n = {int(perf.loc['UPF', 'n_train']):,})\nTesting 30% (n = {int(perf.loc['UPF', 'n_test']):,})\n"
        f"r = {perf.loc[EXPOSURES, 'r_test'].min():.2f}–{perf.loc[EXPOSURES, 'r_test'].max():.2f}\n"
        "+ competing BMI signature", "#f1f8ef")
    box(36, 2, 28, 22, "Incident digestive disease",
        f"Independent Olink participants\nn = {sig_n:,} (training set excluded)\n"
        "IBD (CD, UC), gallstone disease,\nacute pancreatitis (all and\nnon-gallstone)\n"
        "Positive controls: MASLD, cirrhosis\n5 negative-control outcomes", "#f6eef8")
    box(71, 2, 28, 22, "Specificity & mechanism",
        "M1: + kidney function, CRP\nM2: + waist, ALT, GGT, HbA1c, TG\nM3: + BMI proteomic signature\n"
        "Mediation via intestinal\nepithelial proteins", "#f4f4f4")
    box(1, 2, 28, 22, "Self-reported diet",
        f"Diet scores → outcomes\nn = {flow['24h recall >=1 (whole cohort)']:,}\nsame outcomes and models", "#eef4fb")
    arrow(29.5, 39, 35.5, 39); arrow(64.5, 39, 70.5, 39); arrow(85, 29.5, 64.5, 18)
    arrow(15, 29.5, 15, 24.5); arrow(64.5, 13, 70.5, 13)
    save(fig, "Fig1_design")


# ---------------------------------------------------------------- Fig 2 volcano
def fig2():
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.4), sharey=False)
    for ax, e, tag in zip(axes, EXPOSURES, "abcd"):
        r = pd.read_csv(RES / f"diet_proteome_{e}_M1.csv")
        r["y"] = -np.log10(r.p.clip(lower=1e-300))
        thr = -np.log10(0.05 / len(r))
        ax.axhline(thr, color="#888888", lw=0.5, ls="--", zorder=0)
        ax.axvline(0, color="#888888", lw=0.5, zorder=0)
        ns, neg, pos = r[r.bonf >= .05], r[(r.bonf < .05) & (r.beta < 0)], r[(r.bonf < .05) & (r.beta > 0)]
        ax.scatter(ns.beta, ns.y, s=3, c=GREY, lw=0, zorder=1)
        ax.scatter(neg.beta, neg.y, s=3, c=BLUE, lw=0, zorder=2)
        ax.scatter(pos.beta, pos.y, s=3, c=RED, lw=0, zorder=2)
        epi = r[r.protein.isin(EPI)]
        ax.scatter(epi.beta, epi.y, s=12, facecolors="none", edgecolors="black", lw=0.6, zorder=3)
        lab = pd.concat([r.head(3), epi[epi.bonf < .05].head(4)]).drop_duplicates("protein").sort_values("y")
        ymax = r.y.max()
        placed = []
        for _, s in lab.iterrows():
            ty = s.y
            for py in placed:          # push labels apart vertically
                if abs(ty - py) < 0.06 * ymax:
                    ty = py + 0.06 * ymax
            placed.append(ty)
            dy = (ty - s.y) / ymax * 115   # data units -> points (approximate axis height)
            ax.annotate(s.protein.upper(), (s.beta, s.y), xytext=(5, dy), textcoords="offset points", fontsize=5.5,
                        ha="left", va="center", fontweight="bold" if s.protein in EPI else "normal",
                        arrowprops=dict(arrowstyle="-", lw=0.3, color="#555555", shrinkA=0.5, shrinkB=1.5,
                                        relpos=(0, 0.5)))
        ax.set_title(f"{SHORT[e]}  (n Bonferroni = {len(neg) + len(pos)})", fontsize=7.5)
        ax.set_xlabel("β per SD of diet (SD protein)")
        ax.text(-0.26, 1.08, tag, transform=ax.transAxes, fontsize=10, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(r"$-\log_{10}$ $P$")
    h = [plt.Line2D([], [], marker="o", ls="", ms=3, color=c) for c in (GREY, BLUE, RED)] + \
        [plt.Line2D([], [], marker="o", ls="", ms=4, mfc="none", mec="black")]
    fig.legend(h, ["Bonferroni ≥ 0.05", "Inverse association", "Positive association", "Intestinal epithelial protein"],
               loc="lower center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    save(fig, "Fig2_diet_proteome_volcano")


# ---------------------------------------------------------------- Fig 3 forest
def fig3():
    r = pd.read_csv(RES / "cox_signatures.csv")
    outs = PRIMARY + POSITIVE + NEGATIVE
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 4.6), sharey=True)
    ypos = {o: i for i, o in enumerate(reversed(outs))}
    for ax, e, tag in zip(axes, EXPOSURES, "abcd"):
        for m, off, col in [("M1", 0.17, "#1a1a1a"), ("M3", -0.17, "#d97706")]:
            t = r[(r.model == m) & (r.x == f"sig_{e}") & r.outcome.isin(outs)]
            y = t.outcome.map(ypos) + off
            ax.errorbar(t.hr, y, xerr=[t.hr - t.lci, t.uci - t.hr], fmt="o", ms=2.8, color=col, elinewidth=0.7,
                        capsize=0, label=m)
        ax.axvline(1, color="#888888", lw=0.5, zorder=0)
        for b in [len(NEGATIVE) - 0.5, len(NEGATIVE) + len(POSITIVE) - 0.5]:
            ax.axhline(b, color="#dddddd", lw=0.6)
        ax.set_xscale("log"); ax.set_xticks([0.5, 0.7, 1, 1.4, 2]); ax.set_xticklabels(["0.5", "0.7", "1", "1.4", "2"])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlim(0.45, 2.6)
        ax.set_title(f"{SHORT[e]} signature", fontsize=8)
        ax.set_xlabel("HR per SD (95% CI)")
        ax.text(-0.08 if tag != "a" else -1.25, 1.02, tag, transform=ax.transAxes, fontsize=10, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(list(ypos.values())); axes[0].set_yticklabels([OUT_LABEL[o] for o in ypos])
    hd = [plt.Line2D([], [], marker="o", ms=3, color=c, lw=0.7) for c in ("#1a1a1a", "#d97706")]
    fig.legend(hd, ["Model 1", "Model 3"], loc="lower center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.6, -0.03))
    fig.tight_layout()
    save(fig, "Fig3_signature_outcomes")


# ---------------------------------------------------------------- Fig 4 epithelial proteins
def fig4():
    fr = pd.read_csv(RES / "foodgroup_drivers.csv")
    prots = ["guca2a", "ddc", "fabp2", "fabp6", "cdh17", "dsg2"]
    fr = fr[fr.protein.isin(prots)]
    order = fr[fr.protein == "guca2a"].set_index("food_group").beta.sort_values().index
    mat = fr.pivot(index="food_group", columns="protein", values="beta").loc[order, prots]
    pv = fr.pivot(index="food_group", columns="protein", values="p").loc[order, prots]
    po = pd.read_csv(RES / "cox_epithelial_proteins.csv").query("model == 'M3'")
    med = pd.read_csv(RES / "mediation.csv").query("outcome == 'Gallstones' and mediator == 'guca2a'")
    fig = plt.figure(figsize=(7.2, 3.9))
    ax1 = fig.add_axes([0.12, 0.13, 0.23, 0.8])
    v = np.abs(mat.values).max()
    im = ax1.imshow(mat.values, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if pv.values[i, j] < 0.05 / pv.size:
                ax1.text(j, i, "•", ha="center", va="center", fontsize=7)
    ax1.set_xticks(range(len(prots))); ax1.set_xticklabels([p.upper() for p in prots], rotation=45, ha="right", fontsize=6.5)
    ax1.set_yticks(range(len(order))); ax1.set_yticklabels([FG_LABEL.get(o, o.replace("_", " ").capitalize()) for o in order], fontsize=6.5)
    cax = fig.add_axes([0.365, 0.35, 0.01, 0.4])
    cb = fig.colorbar(im, cax=cax); cb.set_label("β per SD log intake", fontsize=6)
    cb.ax.tick_params(labelsize=6)
    ax1.text(-0.62, 1.03, "a", transform=ax1.transAxes, fontsize=10, fontweight="bold")
    # b: proteins -> outcomes
    ax2 = fig.add_axes([0.6, 0.13, 0.15, 0.8])
    outs = ["Gallstones", "IBD", "Crohn", "UC", "Acute_pancreatitis"]
    cols = {"guca2a": "#1a1a1a", "fabp6": "#2b6cb0", "ddc": "#d97706"}
    for k, (pr, col) in enumerate(cols.items()):
        t = po[(po.protein == pr) & po.outcome.isin(outs)].set_index("outcome").loc[outs[::-1]]
        y = np.arange(len(outs)) + (k - 1) * 0.22
        ax2.errorbar(t.hr, y, xerr=[t.hr - t.lci, t.uci - t.hr], fmt="o", ms=2.6, color=col, elinewidth=0.7, label=pr.upper())
    ax2.axvline(1, color="#888888", lw=0.5, zorder=0)
    ax2.set_yticks(range(len(outs))); ax2.set_yticklabels([OUT_LABEL[o] for o in outs[::-1]], fontsize=6.5)
    ax2.set_xscale("log"); ax2.set_xticks([0.6, 0.8, 1, 1.25]); ax2.set_xticklabels(["0.6", "0.8", "1", "1.25"])
    ax2.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax2.set_xlabel("HR per SD protein (95% CI)", fontsize=7)
    ax2.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.35, 1.09), ncol=3, handletextpad=0.1,
               columnspacing=0.6)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.text(-1.3, 1.03, "b", transform=ax2.transAxes, fontsize=10, fontweight="bold")
    # c: indirect effect via GUCA2A
    ax3 = fig.add_axes([0.85, 0.13, 0.13, 0.8])
    med = med.set_index("exposure").loc[EXPOSURES[::-1]]
    y = np.arange(len(med))
    ax3.errorbar(med.indirect_HR, y, xerr=[med.indirect_HR - med.indirect_lci, med.indirect_uci - med.indirect_HR],
                 fmt="o", ms=3, color="#1a1a1a", elinewidth=0.7)
    ax3.axvline(1, color="#888888", lw=0.5, zorder=0)
    ax3.set_yticks(y); ax3.set_yticklabels([SHORT[e] for e in med.index], fontsize=6.5)
    ax3.set_xlabel("Indirect HR via GUCA2A\n(gallstone disease)", fontsize=7)
    ax3.tick_params(axis="x", labelsize=6)
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.text(-0.75, 1.03, "c", transform=ax3.transAxes, fontsize=10, fontweight="bold")
    save(fig, "Fig4_epithelial_proteins")



# ---------------------------------------------------------------- Supplementary Fig 1 dose-response splines
def figS1():
    c = pd.read_csv(RES / "tier1_spline_curves.csv")
    outs = ["Gallstones", "IBD", "Acute_pancreatitis"]
    fig, axes = plt.subplots(3, 4, figsize=(7.2, 5.6), sharex=True)
    for i, o in enumerate(outs):
        for j, e in enumerate(EXPOSURES):
            ax = axes[i, j]
            t = c[(c.outcome == o) & (c.x == f"sig_{e}")]
            ax.fill_between(t.value, t.lci, t.uci, color="#9ecae1", lw=0)
            ax.plot(t.value, t.hr, color="#08519c", lw=1)
            ax.axhline(1, color="#888888", lw=0.5, ls="--")
            ax.set_yscale("log")
            lo, hi = t.lci.min(), t.uci.max()
            ticks = [v for v in [0.2, 0.3, 0.5, 0.7, 1, 1.4, 2, 3, 5] if lo * 0.9 <= v <= hi * 1.1]
            ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
            ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.tick_params(labelsize=6.5)
            ax.spines[["top", "right"]].set_visible(False)
            ax.text(0.03, 0.95, rf"$P$ nonlinear = {t.p_nonlinear.iloc[0]:.2f}", transform=ax.transAxes, fontsize=6, va="top")
            if i == 0:
                ax.set_title(f"{SHORT[e]} signature", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{OUT_LABEL[o]}\nHR (95% CI)", fontsize=7)
            if i == 2:
                ax.set_xlabel("Signature (SD)", fontsize=7)
    fig.tight_layout()
    save(fig, "FigS1_dose_response_splines")

if __name__ == "__main__":
    import sys
    for f in (sys.argv[1:] or ["fig1", "fig2", "fig3", "fig4"]):
        globals()[f]()
        print("done", f)
