# Player Re-Identification in a Single Feed

## Overview

The goal of this system is to reliably attach IDs to players and consistently maintain the same ID as they move around the field, go for the ball, etc. Beyond in-frame tracking, the system attempts to re-identify players that leave the camera view and re-enter, attaching them the same ID that was originally assigned.

---

## Project Structure

```
project/
    run_tracking.py       # Main Script
    track.py              # track state
    association.py        # Hungarian matching with motion gating
    embedder_reid.py      # OSNet-AIN appearance embeddings
    utils.py            
    models/
        best.pt           # YOLO player detector
    assets/
        15sec_input_720p.mp4   # input video (place here)
    requirements.txt
```

---

## Setup

**1. Create and activate a virtual environment**
```bash
python -m venv env
# Then activate as per your OS (mac or windows)
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run**
```bash
python run_tracking.py
```

Optionally, if you want to add a video in a different location (dont want to make an assets folder)
```bash
# Be sure to save your yolo detection model in the models folder for default running or put in the correct location
python run_tracking.py --video assets/15sec_input_720p.mp4 --model models/best.pt
```

**Outputs:**
- `tracked_output.mp4` — annotated video with player IDs

---

## Approach & Methodology

This system was built from scratch using independent functions rather than dropping in an existing tracking library like DeepSORT. The reasoning behind this was to understand the complexity behind the problem — building something from the base before using well-established or SOTA methods helps understand why those methods are important, what changes made them superior, and gives more meaning and interpretability to the problem.

The pipeline follows these steps each frame:

```
YOLO detection
↓
Full bbox crop per player
↓
OSNet-AIN appearance embedding
↓
Motion-gated Hungarian matching → existing tracks
↓
Track update / new track creation
↓
Output IDs + annotated frame
```

**Appearance Embedding — OSNet-AIN**

OSNet-AIN was chosen over more generic models like CLIP or ResNet because OSNet is trained specifically for Re-ID tasks. CLIP handles images on the whole — it tries to make a general description of what it sees — whereas OSNet embeds more fine-grained meaning that makes Re-ID more reliable. The AIN (Adaptive Instance Normalisation) variant was specifically chosen because it adds normalisation that mitigates lighting distortion and slight occlusion better than the base OSNet, both of which are common in broadcast football footage.

**Motion Gating**

Pure visual matching alone felt unreliable — players on opposite teams can look near identical in embeddings, and a player exiting one side of the frame could be incorrectly matched to a similar-looking player appearing on the opposite side. To reduce these false matches, a motion neighbourhood constraint was added: before computing appearance similarity, detections that are physically too far from a track's predicted position are rejected outright (inspired by kalmann filter position gating). This significantly reduces poor matches for players actively on the field.

**Hungarian Assignment**

Matching is done via the Hungarian algorithm rather than greedy assignment. This finds the globally optimal assignment across all track-detection pairs simultaneously, which matters when multiple players are close together and their similarity scores are competitive.

**Persistent IDs**

Tracks never expire. Once a player is assigned an ID it is kept for the full duration of the video, so that if they re-enter the frame the appearance matching has a stored embedding to compare against.

---

## Techniques Tried & Outcomes

**ResNet18 (torchvision)** — first attempt. Reasonable for continuous tracking but struggled with pose changes.

**CLIP (ViT-B/32)** — tried as a richer feature extractor. Found it was relatively poor at handling pose change and appearance variation. The issue is that CLIP tries to describe the image holistically rather than embed fine-grained identity features — not the right tool for distinguishing two players in matching kits.

**Top-60% crop** — explicitly cropping only the upper body (torso and head) to reduce distortion caused by different leg positions. The torso and head are more stable identity anchors than legs and feet. This made a noticeable difference to in-frame ID consistency.

**OSNet-AIN (current)** — the best result achieved. Relatively decent at keeping consistent IDs for players on the field at any given moment, though Re-ID of re-entering players still falls short of reliable.

---

## Challenges

The most persistent issue was ID swaps between players that were still within the field — not even a Re-ID problem, just in-frame switches. This is what led to the top-half cropping approach, which made sense because it reduces the noise that pose variation introduces into the embedding. Legs and feet vary dramatically with motion and add noise rather than identity signal.

Re-identification of players re-entering the frame remains the weakest part of the system. The core difficulty is that the stored embeddings are computed from small broadcast-distance crops (~30–60px wide), where colour, texture, and jersey number details are heavily degraded.

---

## What Remains & Next Steps

The system is functional but incomplete with respect to reliable re-entry Re-ID. With more time and compute, the following would be the clear upgrade path:

**1. Domain-specific Re-ID model**
The single biggest lever would be replacing OSNet with a model fine-tuned on the SoccerNet Re-Identification dataset — broadcast football footage, matching kits, small crops, the exact scenario of this task. The `shallowlearn/sportsreid` repository (https://github.com/shallowlearn/sportsreid) provides ResNet50-fc512 checkpoints trained on SoccerNet-ReID. With a GPU and no size constraint this would be the immediate next step.

**2. Jersey number**
Jersey numbers are unique per player and robust to pose change. Running a lightweight OCR model (e.g. PaddleOCR) on each crop alongside the embedding model would provide a definitive identity signal when the number is legible, with appearance embeddings as a fallback. This would be a nice attempt at trying to add more meaning to the detection and make ID assigning more fact-based

**3. SOTA tracking methods**
Methods like **BoT-SORT** and **StrongSORT** extend the DeepSORT framework with better Re-ID integration and camera motion compensation — directly relevant to a moving broadcast camera. **RF-DETR** would replace YOLO as the detector, potentially improving detection quality on small/occluded players, though it does not directly address the Re-ID component. Benchmarking these against the current pipeline would give a clear picture of where the gains come from.

**4. Performance and Latency**
A good amount of my time went into trying to solve the problem, and a little less into making the most optimal approach that performs the best in terms of the latency-accuracy tradeoffs etc. The current approach is lightweight enough to run on a CPU in a decent amount of time (albeit not close to a real-time solution) and I would love to explore more ways I can improve the speed of the system after making it work satisfactorily with an accuracy/performance I'd ideally desire.

I'd also like to try exporting the model and using it in the ONNX runtime, rather than a direct .pth/.pt file to see if and how significantly it impacts the speed of the system.