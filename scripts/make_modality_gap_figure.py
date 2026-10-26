"""Copy the existing modality-gap plot into the paper figures folder.

If runs/diag/ubfc_modality_gap.png exists, copy it to
paper/figures/modality_gap.png (and .pdf via matplotlib re-save).
"""
import os
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

os.makedirs("paper/figures", exist_ok=True)
src = "runs/diag/ubfc_modality_gap.png"
if not os.path.exists(src):
    print(f"[FAIL] {src} not found — run step7 first")
    raise SystemExit(1)

# Copy PNG
shutil.copy(src, "paper/figures/modality_gap.png")

# Re-save as PDF via matplotlib so pdflatex has a vector-ish asset
img = mpimg.imread(src)
fig = plt.figure(figsize=(7.0, 2.6))
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(img)
ax.axis("off")
fig.savefig("paper/figures/modality_gap.pdf", bbox_inches="tight")
plt.close(fig)
print("[ OK ] wrote paper/figures/modality_gap.{png,pdf}")