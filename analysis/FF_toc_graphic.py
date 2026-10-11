"""Table-of-contents graphic for Food & Function: 8 cm x 4 cm, 600 dpi RGB TIFF (+ PNG preview), Arial.
Diet quality -> intestinal epithelial proteome -> digestive disease, with the headline hazard ratios.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

from common import PROJ

OUT = PROJ / "manuscript_FoodFunction" / "figures"
plt.rcParams.update({"font.family": "Arial", "font.size": 6})
W, H = 8 / 2.54, 4 / 2.54
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 80); ax.set_ylim(0, 40); ax.axis("off")


def box(x, y, w, h, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.5", fc=fc, ec="#4a4a4a", lw=0.5))


def arrow(x1, x2, y):
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>", mutation_scale=7, lw=0.8, color="#4a4a4a"))


# panel 1: diet
box(1.5, 3, 19, 34, "#eef4fb")
ax.text(11, 34, "Diet quality", ha="center", va="top", fontsize=7, fontweight="bold")
ax.text(11, 27.5, "Healthy plants\n& fibre", ha="center", va="top", fontsize=6, color="#2b6cb0", linespacing=1.2)
ax.text(11, 16.5, "Ultra-processed\n& unhealthy plant\nfoods", ha="center", va="top", fontsize=6, color="#c53030",
        linespacing=1.2)
ax.text(11, 5.0, "n = 204,509", ha="center", va="bottom", fontsize=5.5, color="#4a4a4a")
arrow(21.3, 24.7, 20)

# panel 2: proteome
box(25.5, 3, 21, 34, "#fdf3e7")
ax.text(36, 34, "Plasma proteome", ha="center", va="top", fontsize=7, fontweight="bold")
ax.text(36, 27.5, "2,919 proteins", ha="center", va="top", fontsize=6)
ax.text(36, 21.5, "Intestinal\nepithelial imprint", ha="center", va="top", fontsize=6, linespacing=1.2)
ax.text(36, 12.0, "GUCA2A (guanylin)\nDDC · FABP2 · CDH17", ha="center", va="top", fontsize=5.5, fontweight="bold",
        linespacing=1.3)
arrow(47.3, 50.7, 20)

# panel 3: outcomes
box(51.5, 3, 27, 34, "#f1f8ef")
ax.text(65, 34, "Signature → risk", ha="center", va="top", fontsize=7, fontweight="bold")
ax.text(65, 28.5, "HR per SD signature", ha="center", va="top", fontsize=5.5, color="#4a4a4a")
rows = [("Gallstones", "UPF 1.25", "#c53030"), ("IBD", "hPDI 0.71", "#2b6cb0"),
        ("Pancreatitis", "Fibre 0.78", "#2b6cb0")]
for k, (lab, hr, col) in enumerate(rows):
    y = 21.0 - 6.5 * k
    ax.text(52.8, y, lab, ha="left", va="center", fontsize=6)
    ax.text(77.8, y, hr, ha="right", va="center", fontsize=6, fontweight="bold", color=col)
ax.text(65, 3.8, "beyond adiposity", ha="center", va="bottom", fontsize=5.5, color="#4a4a4a", style="italic")

for ext in ("png", "pdf"):
    fig.savefig(OUT / f"TOC_graphic.{ext}", dpi=600)
fig.savefig(OUT / "TOC_rgba.tiff", dpi=600)
im = Image.open(OUT / "TOC_rgba.tiff").convert("RGB")
im.save(OUT / "TOC_graphic.tiff", dpi=(600, 600), compression="tiff_lzw")
(OUT / "TOC_rgba.tiff").unlink()
print("TOC graphic", im.size, f"= {im.size[0] / 600 * 2.54:.2f} x {im.size[1] / 600 * 2.54:.2f} cm")
