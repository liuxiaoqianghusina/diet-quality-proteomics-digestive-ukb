"""Food & Function versions of the figures: same plots as F07_figures.py, redrawn so that the saved image is at most
17.1 cm wide (RSC double-column width) with the font sizes unchanged, then written as RGB, LZW-compressed 600 dpi TIFF
(plus PNG previews). Output: manuscript_FoodFunction/figures/.
Usage: python3 FF_figures.py [fig1 fig2 fig3 fig4 figS1]
"""
import re
import sys

import F07_figures as F
from PIL import Image

from common import PROJ

OUT = PROJ / "manuscript_FoodFunction" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
MAX_W = 17.0 / 2.54          # inches (a little under 17.1 cm to absorb rounding)
PAD = 0.05                   # pad_inches used when saving
NAMES = {"Fig1_design": "Fig1", "Fig2_diet_proteome_volcano": "Fig2", "Fig3_signature_outcomes": "Fig3",
         "Fig4_epithelial_proteins": "Fig4", "FigS1_dose_response_splines": "FigS1"}


def fit_width(fig):
    """Shrink the canvas until the tight bounding box (plus padding) fits within MAX_W; fonts keep their size."""
    for _ in range(8):
        if fig.get_layout_engine() is not None or getattr(fig, "_ff_tight", False):
            fig.tight_layout()
        fig.canvas.draw()
        w = fig.get_tightbbox(fig.canvas.get_renderer()).width + 2 * PAD
        if w <= MAX_W + 1e-3:
            return w
        W, H = fig.get_size_inches()
        fig.set_size_inches(W - (w - MAX_W) - 0.01, H)
    return w


def tweak_fig2(fig):
    """Shorter panel titles so they clear the panel letters at 17 cm width."""
    for ax in fig.axes:
        m = re.match(r"(\S+)\s+\(n Bonferroni = (\d+)\)", ax.get_title())
        if m:
            ax.set_title(f"{m[1]} ({m[2]} proteins)", fontsize=7.5)


def tweak_fig4(fig):
    """Wrap the long outcome label and move the panel b legend off panel c."""
    ax1, cax, ax2, ax3 = fig.axes
    ax2.set_yticks(ax2.get_yticks(), [t.get_text().replace("Inflammatory bowel disease", "Inflammatory\nbowel disease")
                                      for t in ax2.get_yticklabels()], fontsize=6.5)
    ax2.set_position([0.63, 0.13, 0.14, 0.8])
    leg = ax2.get_legend()
    leg.set_bbox_to_anchor((-0.25, 1.09), transform=ax2.transAxes)


TWEAK = {"Fig2_diet_proteome_volcano": tweak_fig2, "Fig4_epithelial_proteins": tweak_fig4}


def save(fig, name):
    if name in TWEAK:
        TWEAK[name](fig)
    w = fit_width(fig)
    stem = NAMES[name]
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=PAD)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=PAD)
    tmp = OUT / f"{stem}_rgba.tiff"
    fig.savefig(tmp, dpi=600, bbox_inches="tight", pad_inches=PAD)
    im = Image.open(tmp).convert("RGB")
    im.save(OUT / f"{stem}.tiff", dpi=(600, 600), compression="tiff_lzw")
    tmp.unlink()
    print(f"{stem}: {im.size[0] / 600 * 2.54:.2f} x {im.size[1] / 600 * 2.54:.2f} cm (tight width {w * 2.54:.2f} cm)")
    F.plt.close(fig)


# fig2, fig3 and figS1 call fig.tight_layout() before saving; re-apply it after each resize
_orig_tl = F.plt.Figure.tight_layout


def _tl(self, *a, **k):
    self._ff_tight = True
    return _orig_tl(self, *a, **k)


F.plt.Figure.tight_layout = _tl
F.save = save
for f in (sys.argv[1:] or ["fig1", "fig2", "fig3", "fig4", "figS1"]):
    getattr(F, f)()
