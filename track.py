# Helps in making the track dictionary for tracking individual players
def new_track(track_id, bbox, embedding, frame_id):
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    return {
        "id": track_id,
        "bbox": bbox,
        "embedding": embedding,
        "velocity": (0.0, 0.0),
        "center": (cx, cy),
        "last_seen": frame_id,
    }


def update_track(track, bbox, embedding, frame_id):
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    vx = cx - track["center"][0]
    vy = cy - track["center"][1]
    track["bbox"] = bbox
    track["velocity"] = (vx, vy)
    track["center"] = (cx, cy)
    track["last_seen"] = frame_id
    # Exponential moving average on embedding
    track["embedding"] = 0.8 * track["embedding"] + 0.2 * embedding
    return track
