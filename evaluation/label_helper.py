"""Build face ground truth for the evaluation video (candidates -> review -> labels).

There are no hand-drawn labels for the sample video, so ground truth is made
in two steps that don't depend on either detector's tuned thresholds:

  1. ``python evaluation/label_helper.py candidates``
     Samples every Nth frame and proposes candidate face boxes from BOTH
     detectors using deliberately loose thresholds (so real faces are very
     unlikely to be missed). Candidates are numbered and rendered into
     review sheets under evaluation/labeling/sheets/.
  2. A person looks at the sheets and records which candidates are NOT faces
     (and any faces nobody found) in evaluation/labeling/decisions.json.
     ``python evaluation/label_helper.py build`` then writes
     evaluation/ground_truth.json.

decisions.json format::

    {"reject_haar_only": true,        # optional: drop candidates the DNN did not find
     "reject": ["<frame>:<candidate id>", ...],
     "add":    [{"frame": 120, "box": [x, y, w, h]}, ...],
     "ignore": [{"frame": 660, "box": [x, y, w, h]}, ...]}   # too small/occluded to label

Replace evaluation/ground_truth.json with your own labels in the same format
to evaluate against different truth.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EVAL_DIR)
sys.path.insert(0, ROOT)

import app  # noqa: E402  (needs ROOT on sys.path)

LABEL_DIR = os.path.join(EVAL_DIR, "labeling")
SHEET_DIR = os.path.join(LABEL_DIR, "sheets")
CANDIDATES_PATH = os.path.join(LABEL_DIR, "candidates.json")
DECISIONS_PATH = os.path.join(LABEL_DIR, "decisions.json")
GROUND_TRUTH_PATH = os.path.join(EVAL_DIR, "ground_truth.json")

CLUSTER_IOU = 0.3


def find_video() -> str:
    path = app.discover_video(app.VIDEOS_DIR)
    if not path:
        sys.exit("No video found in videos/.")
    return path


def cluster(boxes_by_source: list[tuple[str, tuple]]) -> list[dict]:
    """Group boxes from different detectors that overlap into one candidate."""
    clusters = []
    for source, box in boxes_by_source:
        for c in clusters:
            if source not in c["sources"] and app.box_iou(c["members"][0], box) > CLUSTER_IOU:
                c["members"].append(box)
                c["sources"].append(source)
                break
        else:
            clusters.append({"members": [box], "sources": [source]})
    for c in clusters:
        c["box"] = [int(round(v)) for v in np.mean(np.array(c["members"], dtype=float), axis=0)]
        del c["members"]
    return clusters


def make_candidates(step: int) -> None:
    import logging
    logging.disable(logging.INFO)

    dnn_settings = dict(app.DEFAULT_CONFIG["dnn"], score_threshold=0.4)
    dnn = app.load_dnn_detector(dnn_settings)
    alt = app.load_cascade(app.resolve_path("cascades/haarcascade_frontalface_alt.xml"))
    default = app.load_cascade(app.resolve_path("cascades/haarcascade_frontalface_default.xml"))

    video = find_video()
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    os.makedirs(SHEET_DIR, exist_ok=True)

    frames = {}
    tiles = []  # (caption frame, rendered image)
    empty_frames = []
    for index in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        found = [("dnn", tuple(f[:4])) for f in app.detect_faces_dnn(frame, dnn)]
        for name, cascade, mn in (("alt", alt, 2), ("default", default, 3)):
            for rect in cascade.detectMultiScale(gray, 1.05, mn, minSize=(20, 20)):
                found.append((name, tuple(int(v) for v in rect)))
        candidates = cluster(found)
        for i, c in enumerate(candidates):
            c["id"] = i
        frames[str(index)] = candidates

        if not candidates:
            empty_frames.append(index)
            continue
        canvas = frame.copy()
        for c in candidates:
            x, y, w, h = c["box"]
            color = (0, 255, 0) if "dnn" in c["sources"] else (0, 140, 255)
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
            cv2.putText(canvas, str(c["id"]), (x, max(y - 5, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)
        cv2.putText(canvas, "frame %d" % index, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        tiles.append(cv2.resize(canvas, (640, 360)))

    for n in range(0, len(tiles), 4):
        chunk = tiles[n:n + 4]
        while len(chunk) < 4:
            chunk.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(chunk[:2]), np.hstack(chunk[2:])])
        cv2.imwrite(os.path.join(SHEET_DIR, "sheet_%02d.png" % (n // 4)), sheet)

    with open(CANDIDATES_PATH, "w", encoding="utf-8") as f:
        json.dump({"video": os.path.basename(video), "frame_step": step, "frames": frames}, f, indent=1)
    print("frames sampled: %d | with candidates: %d | empty: %d" % (len(frames), len(tiles), len(empty_frames)))
    print("review sheets:", SHEET_DIR)


def build_ground_truth() -> None:
    with open(CANDIDATES_PATH, encoding="utf-8") as f:
        cand = json.load(f)
    with open(DECISIONS_PATH, encoding="utf-8") as f:
        decisions = json.load(f)
    rejected = set(decisions.get("reject", []))
    reject_unconfirmed = decisions.get("reject_haar_only", False)

    truth = {}
    for frame, candidates in cand["frames"].items():
        truth[frame] = [
            c["box"] for c in candidates
            if "%s:%d" % (frame, c["id"]) not in rejected
            and not (reject_unconfirmed and "dnn" not in c["sources"])
        ]
    for extra in decisions.get("add", []):
        truth.setdefault(str(extra["frame"]), []).append(list(extra["box"]))

    ignore = {}
    for region in decisions.get("ignore", []):
        ignore.setdefault(str(region["frame"]), []).append(list(region["box"]))

    out = {
        "video": cand["video"],
        "frame_step": cand["frame_step"],
        "note": (
            "Boxes are the mean of the detectors' boxes for each candidate a person "
            "reviewed and accepted (see label_helper.py). Frames listed with [] are "
            "reviewed and contain no face. ignore_regions mark faces too small or "
            "occluded to label: detections there count as neither right nor wrong."
        ),
        "faces_per_frame": truth,
        "ignore_regions": ignore,
    }
    with open(GROUND_TRUTH_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("ground truth: %d frames, %d faces -> %s" % (
        len(truth), sum(len(v) for v in truth.values()), GROUND_TRUTH_PATH))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("candidates", "build"))
    parser.add_argument("--step", type=int, default=12, help="Sample every Nth frame (default 12).")
    args = parser.parse_args()
    if args.command == "candidates":
        make_candidates(args.step)
    else:
        build_ground_truth()


if __name__ == "__main__":
    main()
