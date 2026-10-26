"""Generate the pipeline overview figure for the paper.

Produces: paper/figures/pipeline.{pdf,png}
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

os.makedirs("paper/figures", exist_ok=True)

fig, ax = plt.subplots(figsize=(7.0, 2.6))
ax.set_xlim(0, 10)
ax.set_ylim(-0.3, 3.0)
ax.axis("off")

def box(x, y, w, h, text, color="#E3F2FD", fontsize=8):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.05,rounding_size=0.08",
                       linewidth=0.7, edgecolor="#333", facecolor=color)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize)

def arrow(x1, y1, x2, y2, color="#333"):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle="-|>", mutation_scale=10,
                        linewidth=0.8, color=color)
    ax.add_patch(a)

# ---------------- Row 1: signal chain ----------------
box(0.05, 1.85, 1.30, 0.85,  "Face\nvideo", "#FFF3E0")
box(1.55, 1.85, 1.55, 0.85,  "FaceMesh\n+ 3 ROIs", "#E3F2FD")
box(3.30, 1.85, 1.70, 0.85,  "SQI-weighted\nROI fusion", "#E3F2FD")
box(5.20, 1.85, 1.10, 0.85,  "POS\nrPPG", "#E3F2FD")
box(6.50, 1.85, 1.60, 0.85,  "IBI extraction\n+ outlier filter", "#E3F2FD")
box(8.30, 1.85, 1.65, 0.85,  "6-D HRV\nfeatures", "#C8E6C9")

# Row 1 horizontal arrows
arrow(1.35, 2.275, 1.55, 2.275)
arrow(3.10, 2.275, 3.30, 2.275)
arrow(5.00, 2.275, 5.20, 2.275)
arrow(6.30, 2.275, 6.50, 2.275)
arrow(8.10, 2.275, 8.30, 2.275)

# ---------------- Row 2: output heads ----------------
# HR and HRV come from IBI extraction
box(1.55, 0.20, 1.55, 0.85,  "HR\n(regression)", "#F3E5F5")
box(3.30, 0.20, 1.70, 0.85,  "HRV\nanalysis",      "#F3E5F5")
# Stress comes from HRV features
box(5.20, 0.20, 2.90, 0.85,  "Stress classifier\n(ECG / PPG / synthetic-rPPG)", "#F3E5F5")
# Emotion is a parallel branch from the face image
box(8.30, 0.20, 1.65, 0.85,  "Emotion head\n(3-class valence)", "#FFEBEE")

# ---------------- Row1 -> Row2 arrows ----------------
# IBI box (middle x = 7.30) -> HR box (middle x = 2.325)
arrow(6.80, 1.85, 2.325, 1.05)
# IBI box -> HRV box (middle x = 4.15)
arrow(6.95, 1.85, 4.15, 1.05)
# HRV features box (middle x = 9.125, top edge y=1.85) -> Stress box (middle x = 6.65)
arrow(9.00, 1.85, 6.65, 1.05)
# Face video box (middle x = 0.70, bottom y=1.85) -> Emotion box (middle x = 9.125)
# Draw as a curved arrow for clarity
from matplotlib.patches import FancyArrowPatch as FAP
curve = FAP((0.70, 1.85), (9.125, 1.05),
            connectionstyle="arc3,rad=-0.35",
            arrowstyle="-|>", mutation_scale=10,
            linewidth=0.8, color="#333", linestyle="--")
ax.add_patch(curve)

plt.tight_layout()
out = "paper/figures/pipeline.pdf"
plt.savefig(out, bbox_inches="tight")
plt.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
plt.close()
print(f"[ OK ] wrote {out} and .png")