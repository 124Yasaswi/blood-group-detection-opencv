"""Create realistic-looking sample slide images, so the app can be tried without a lab.

* Smooth drop  (no agglutination): an even red drop with fine grain.
* Clumped drop (agglutination):    dark red clumps of different sizes with
                                   pale liquid between them.

Every blood group gets a folder with anti_a.png, anti_b.png and anti_d.png.

Usage:
    python -m src.generate_samples            # writes samples/<group>/...
"""
from pathlib import Path

import cv2
import numpy as np

SIZE = 400
GROUPS = {  # group: (Anti-A clumps?, Anti-B clumps?, Anti-D clumps?)
    "A+": (1, 0, 1), "A-": (1, 0, 0), "B+": (0, 1, 1), "B-": (0, 1, 0),
    "AB+": (1, 1, 1), "AB-": (1, 1, 0), "O+": (0, 0, 1), "O-": (0, 0, 0),
}


def _background(rng):
    img = np.full((SIZE, SIZE, 3), (235, 238, 240), np.float32)  # white card (BGR)
    img += rng.normal(0, 3, img.shape)
    return img


def _drop_mask(rng):
    mask = np.zeros((SIZE, SIZE), np.uint8)
    c = (SIZE // 2 + int(rng.integers(-10, 10)), SIZE // 2 + int(rng.integers(-10, 10)))
    axes = (int(rng.integers(140, 160)), int(rng.integers(135, 155)))
    cv2.ellipse(mask, c, axes, float(rng.uniform(0, 180)), 0, 360, 255, -1)
    return cv2.GaussianBlur(mask, (9, 9), 0).astype(np.float32) / 255.0


def smooth_drop(rng) -> np.ndarray:
    img, m = _background(rng), _drop_mask(rng)
    red = np.zeros_like(img)
    red[:] = (40, 45, 175)                       # even blood colour
    red += rng.normal(0, 6, red.shape)            # fine grain
    out = img * (1 - m[..., None]) + red * m[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def clumped_drop(rng) -> np.ndarray:
    img, m = _background(rng), _drop_mask(rng)
    liquid = np.zeros_like(img)
    liquid[:] = (150, 160, 215)                   # pale, cleared liquid
    clumps = np.zeros((SIZE, SIZE), np.uint8)
    for _ in range(int(rng.integers(25, 40))):
        x, y = rng.integers(70, SIZE - 70, 2)
        r = int(rng.integers(6, 20))
        pts = (np.array([x, y]) + rng.normal(0, r * 0.6, (8, 2))).astype(np.int32)
        cv2.fillPoly(clumps, [cv2.convexHull(pts)], 255)
        cv2.circle(clumps, (int(x), int(y)), r, 255, -1)
    c = cv2.GaussianBlur(clumps, (5, 5), 0).astype(np.float32)[..., None] / 255.0
    dark = np.zeros_like(img)
    dark[:] = (25, 25, 120)                       # dense red-brown clumps
    drop = liquid * (1 - c) + dark * c + rng.normal(0, 6, img.shape)
    out = img * (1 - m[..., None]) + drop * m[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def main(out_dir: Path = Path(__file__).resolve().parent.parent / "samples", seed: int = 7):
    rng = np.random.default_rng(seed)
    for group, flags in GROUPS.items():
        folder = out_dir / group.replace("+", "_pos").replace("-", "_neg")
        folder.mkdir(parents=True, exist_ok=True)
        for name, clumped in zip(["anti_a", "anti_b", "anti_d"], flags):
            img = clumped_drop(rng) if clumped else smooth_drop(rng)
            cv2.imwrite(str(folder / f"{name}.png"), img)
    print(f"samples written to {out_dir}")


if __name__ == "__main__":
    main()
