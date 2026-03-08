import argparse
import os

import cv2
from ultralytics import YOLO

from embedder_reid import get_embedding
from association import associate
from track import new_track, update_track
from utils import crop_player, draw_tracks


CONF_THRESHOLD    = 0.7 
MOTION_THRESHOLD  = 140 
SIM_THRESHOLD     = 0.4  
MAX_MISSING       = float("inf")  
OUTPUT_VIDEO      = "tracked_output.mp4"

def run(video_path: str, model_path: str):
    assert os.path.exists(video_path),  f"Video not found: {video_path}"
    assert os.path.exists(model_path),  f"Model not found: {model_path}"

    detector = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    tracks   = []          # list of active track dicts
    next_id  = 1
    frame_id = 0

    print(f"Processing {video_path}  ({width}x{height} @ {fps:.1f} fps)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detector(frame, verbose=False)[0]
        raw_boxes = results.boxes

        detections = []
        for box in raw_boxes:
            conf = float(box.conf)
            if conf < CONF_THRESHOLD:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({"bbox": [x1, y1, x2, y2], "conf": conf})

        for det in detections:
            crop = crop_player(frame, det["bbox"])
            det["embedding"] = get_embedding(crop)

        matches, _, unmatched_dets = associate(
            tracks, detections,
            motion_threshold=MOTION_THRESHOLD,
            similarity_threshold=SIM_THRESHOLD,
        )

        for ti, di in matches:
            update_track(tracks[ti], detections[di]["bbox"],
                         detections[di]["embedding"], frame_id)

        for di in unmatched_dets:
            trk = new_track(next_id,
                            detections[di]["bbox"],
                            detections[di]["embedding"],
                            frame_id)
            tracks.append(trk)
            next_id += 1

        tracks = [t for t in tracks
                  if frame_id - t["last_seen"] <= MAX_MISSING]

        for trk in tracks:
            if trk["last_seen"] == frame_id:     
                x1, y1, x2, y2 = [int(v) for v in trk["bbox"]]

        annotated = draw_tracks(frame, [t for t in tracks
                                        if t["last_seen"] == frame_id])
        writer.write(annotated)

        if frame_id % 30 == 0:
            print(f"  CurFrame: {frame_id:4d}  |  Dets: {len(detections):2d}"
                  f" Unique IDs: {next_id-1}")

        frame_id += 1

    cap.release()
    writer.release()

    print(f"\n{frame_id} frames processed, {next_id-1} unique IDs")
    print(f"  Video  → {OUTPUT_VIDEO}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="assets/15sec_input_720p.mp4")
    parser.add_argument("--model", default="models/best.pt")
    args = parser.parse_args()
    run(args.video, args.model)
