"""Detect agglutination in a blood-slide image and work out the blood group.

How the slide test works
------------------------
Three drops of the same blood are mixed with three reagents:
  * Anti-A  -> clumps (agglutinates) if the blood has A antigen
  * Anti-B  -> clumps if the blood has B antigen
  * Anti-D  -> clumps if the blood is Rh positive

| Anti-A | Anti-B | Group |
|--------|--------|-------|
|  yes   |   no   |   A   |
|   no   |  yes   |   B   |
|  yes   |  yes   |  AB   |
|   no   |   no   |   O   |
plus "+" if Anti-D clumps, "-" if it does not.

Image pipeline (one drop)
-------------------------
1. Green plane: red cells absorb green light, so cells look dark and the
   background looks bright in the green channel. Best contrast of R, G, B.
2. Drop mask: blur, threshold, keep the largest blob = the drop.
3. Inside the drop, Otsu threshold separates dark cell clumps from the
   lighter liquid between them.
4. Morphological opening removes speckle noise; closing joins broken clumps.
5. Connected-component analysis counts the clumps and measures them.
6. Decision: an agglutinated drop has several clear clumps AND a large
   brightness variation; a smooth drop has neither.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Thresholds:
    min_clump_area: int = 40       # pixels; smaller blobs are noise
    min_clumps: int = 4            # at least this many clumps
    min_variation: float = 0.24    # std / mean of green values inside the drop
    min_gap_ratio: float = 0.12    # share of the drop that is clear liquid between clumps


@dataclass
class DropResult:
    agglutinated: bool
    clumps: int
    variation: float
    gap_ratio: float
    masks: dict = field(default_factory=dict, repr=False)  # intermediate images for the GUI


def green_plane(bgr: np.ndarray) -> np.ndarray:
    return bgr[:, :, 1]


def drop_mask(green: np.ndarray) -> np.ndarray:
    """Binary mask (255 = drop) of the largest dark region in the image."""
    blur = cv2.GaussianBlur(green, (31, 31), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return np.full_like(green, 255)  # no clear drop: use the whole image
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = np.where(labels == largest, 255, 0).astype(np.uint8)
    # shrink a little so the bright rim of the drop does not count as gaps
    return cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))


def analyse_drop(bgr: np.ndarray, t: Thresholds = Thresholds()) -> DropResult:
    green = green_plane(bgr)
    mask = drop_mask(green)
    inside = green[mask > 0]
    if inside.size == 0:
        raise ValueError("could not find the drop in the image")

    variation = float(inside.std() / max(inside.mean(), 1.0))

    # Otsu threshold computed only on pixels inside the drop
    otsu, _ = cv2.threshold(inside.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    clumps = np.where((green < otsu) & (mask > 0), 255, 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean = cv2.morphologyEx(clumps, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

    n, _, stats, _ = cv2.connectedComponentsWithStats(clean)
    areas = stats[1:, cv2.CC_STAT_AREA]
    count = int((areas >= t.min_clump_area).sum())
    gap_ratio = 1.0 - float((clean > 0).sum()) / float((mask > 0).sum())

    agglutinated = count >= t.min_clumps and variation >= t.min_variation and gap_ratio >= t.min_gap_ratio
    return DropResult(agglutinated, count, round(variation, 3), round(gap_ratio, 3),
                      masks={"green": green, "drop": mask, "clumps": clean})


def blood_group(anti_a: bool, anti_b: bool, anti_d: bool) -> str:
    abo = {(True, False): "A", (False, True): "B", (True, True): "AB", (False, False): "O"}[(anti_a, anti_b)]
    return abo + ("+" if anti_d else "-")


def detect(anti_a_img: np.ndarray, anti_b_img: np.ndarray, anti_d_img: np.ndarray, t: Thresholds = Thresholds()):
    """Return (blood group, {reagent: DropResult})."""
    results = {name: analyse_drop(img, t) for name, img in
               [("Anti-A", anti_a_img), ("Anti-B", anti_b_img), ("Anti-D", anti_d_img)]}
    group = blood_group(results["Anti-A"].agglutinated, results["Anti-B"].agglutinated, results["Anti-D"].agglutinated)
    return group, results
