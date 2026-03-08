import numpy as np
from scipy.optimize import linear_sum_assignment


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)


def _center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def associate(tracks, detections,
              motion_threshold=150,
              similarity_threshold=0.3,
              motion_gate_window=5):
    
    # tracks     : list of track dicts  
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))

    current_frame = max(t["last_seen"] for t in tracks)

    n_trk = len(tracks)
    n_det = len(detections)
    cost = np.ones((n_trk, n_det), dtype=np.float32)  # default: no match

    for ti, trk in enumerate(tracks):
        tcx, tcy = trk["center"]
        vx, vy = trk["velocity"]
        pred_cx = tcx + vx
        pred_cy = tcy + vy

        frames_missing = current_frame - trk["last_seen"]
        apply_motion_gate = frames_missing < motion_gate_window

        for di, det in enumerate(detections):
            if apply_motion_gate:
                dcx, dcy = _center(det["bbox"])
                dist = ((pred_cx - dcx) ** 2 + (pred_cy - dcy) ** 2) ** 0.5
                if dist > motion_threshold:
                    continue  

            sim = _cosine_sim(trk["embedding"], det["embedding"])
            cost[ti, di] = 1.0 - sim   

    row_ind, col_ind = linear_sum_assignment(cost)

    matches = []
    used_tracks = set()
    used_dets   = set()

    for ti, di in zip(row_ind, col_ind):
        if cost[ti, di] >= 1.0 - similarity_threshold:
            # Beyond threshold, caused by failing motion gating or just not matching
            continue
        matches.append((ti, di))
        used_tracks.add(ti)
        used_dets.add(di)

    unmatched_tracks = [i for i in range(n_trk) if i not in used_tracks]
    unmatched_dets   = [i for i in range(n_det) if i not in used_dets]

    return matches, unmatched_tracks, unmatched_dets
