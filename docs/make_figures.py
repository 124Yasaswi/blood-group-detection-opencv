"""Make the README images: docs/pipeline.png and docs/app.png.

Usage (from the project root):
    python docs/make_figures.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import ImageGrab  # noqa: E402

from src.detector import analyse_drop  # noqa: E402

DOCS = ROOT / "docs"


def pipeline_figure():
    rows = [("Anti-A on A+ blood (clumps)", ROOT / "samples/A_pos/anti_a.png"),
            ("Anti-B on A+ blood (smooth)", ROOT / "samples/A_pos/anti_b.png")]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.4))
    for r, (title, path) in enumerate(rows):
        img = cv2.imread(str(path))
        res = analyse_drop(img)
        panels = [(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "1. Input", None),
                  (res.masks["green"], "2. Green plane", "gray"),
                  (res.masks["drop"], "3. Drop mask", "gray"),
                  (res.masks["clumps"], f"4. Clumps: {res.clumps}", "gray")]
        for c, (im, label, cmap) in enumerate(panels):
            axes[r, c].imshow(im, cmap=cmap)
            axes[r, c].set_title(label, fontsize=10)
            axes[r, c].axis("off")
        verdict = "AGGLUTINATED" if res.agglutinated else "no agglutination"
        axes[r, 0].text(-0.08, 0.5, f"{title}\n→ {verdict}", transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(DOCS / "pipeline.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def app_screenshot(sample="AB_neg"):
    from app import App, REAGENTS
    app = App()
    for reagent in REAGENTS:
        app.set_image(reagent, ROOT / "samples" / sample / f"{reagent.lower().replace('-', '_')}.png")
    app.run()
    app.update()
    app.lift()
    app.attributes("-topmost", True)
    app.after(800, lambda: _grab(app))
    app.mainloop()


def _grab(app):
    app.update()
    x, y = app.winfo_rootx(), app.winfo_rooty()
    w, h = app.winfo_width(), app.winfo_height()
    ImageGrab.grab(bbox=(x, y, x + w, y + h)).save(DOCS / "app.png")
    app.destroy()


if __name__ == "__main__":
    pipeline_figure()
    app_screenshot()
    print("figures written to", DOCS)
