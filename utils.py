"""
utils.py - Small helpers used across the pipeline.
"""

import cv2
import numpy as np


def crop_player(frame: np.ndarray, bbox, shrink=True) -> np.ndarray:
    """
    Extract the upper 60% of a player bbox for embedding — jersey and torso
    are far more discriminative than legs/feet for Re-ID.
    Optionally shrink horizontally to reduce background bleed.

    NOTE: this crop is ONLY used for appearance embeddings.
          The full bbox is kept separately for drawing and motion tracking.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]

    # Keep only the top 60% of the box height
    bh = y2 - y1
    y2 = y1 + int(0.70 * bh)

    if shrink:
        bw = x2 - x1
        # x1 = max(0, x1 + int(0.05 * bw))
        # x2 = min(w, x2 - int(0.05 * bw))
        # y1 = max(0, y1 + int(0.10 * bh))   # small top trim (helmet/hair noise)
        # y2 already set above — no bottom trim needed
        x1 = max(0, x1)
        x2 = min(w, x2)
        y1 = max(0, y1)

    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else np.zeros((64, 32, 3), dtype=np.uint8)


# Distinct colours per ID (cycles after 20)
_PALETTE = [
    (255, 56,  56),  (255, 157,  51), (255, 255,  51), ( 51, 255,  51),
    ( 51, 255, 255), ( 51,  51, 255), (255,  51, 255), (255, 153, 153),
    (153, 255, 153), (153, 153, 255), (255, 204, 153), (204, 255, 153),
    (153, 204, 255), (255, 153, 204), (204, 153, 255), (255, 255, 153),
    (153, 255, 255), (255, 153, 255), (200, 200, 200), (100, 100, 100),
]


def draw_tracks(frame: np.ndarray, tracks) -> np.ndarray:
    """Draw bounding boxes and IDs onto frame (in-place copy)."""
    out = frame.copy()
    for trk in tracks:
        x1, y1, x2, y2 = [int(v) for v in trk["bbox"]]
        color = _PALETTE[trk["id"] % len(_PALETTE)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"ID {trk['id']}"
        cv2.putText(out, label, (x1, max(y1 - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out
