"""Face and eye detection with a switchable detector (Haar cascade or DNN).

The detector ("haar" or "dnn"), the cascades, and the video source are
controlled by config.json (paths are relative to this file unless
absolute). CLI flags can override individual config values for one-off
runs. Falls back to the system camera when no usable video file is
configured/found. Designed to run unmodified on a resource-constrained
edge device (headless, no video file, etc.).
"""

import argparse
import json
import logging
import os
import sys
import time

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("face_detection")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(SCRIPT_DIR, "videos")
DEFAULT_CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv")

# Cycled through in order for each enabled cascade, so detections from
# different cascades are visually distinguishable.
BOX_COLOR_PALETTE = [
    (0, 255, 0),    # green
    (0, 165, 255),  # orange
    (255, 0, 0),    # blue
    (255, 0, 255),  # magenta
    (0, 255, 255),  # yellow
]

MERGE_IOU = 0.3  # boxes from different cascades above this overlap are the same face
EYE_UPSCALE = 2  # the eye band is enlarged by this factor before the eye cascade runs
EYE_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
# Where an eye's center may lie, as fractions of the face box: x per side
# (left, right), y from the top, and eye width relative to face width.
EYE_ZONE_X = ((0.10, 0.48), (0.52, 0.90))
EYE_ZONE_Y = (0.22, 0.56)
EYE_SIZE = (0.12, 0.42)
EYE_MIN_SEPARATION = 0.32  # the two eye centers must be this far apart, as a fraction of face width
STACK_ALIGN = 1.0  # a box this many face-widths (or less) off-center below a head is a body box
EYE_BOX_COLOR = (0, 0, 255)  # red, so eyes stand out from any face box color
DNN_BOX_COLOR = BOX_COLOR_PALETTE[0]

VALID_VIDEO_SOURCES = ("auto", "camera")
VALID_DETECTORS = ("haar", "dnn")
VALID_EYE_SOURCES = ("auto", "cascade", "landmarks")

DEFAULT_CONFIG = {
    "video": {"path": "", "camera_index": 0, "source": "auto"},
    "detector": "haar",
    "cascades": [
        {"name": "frontal_alt", "path": "cascades/haarcascade_frontalface_alt.xml", "enabled": True,
         "min_neighbors": 3},
        {"name": "frontal_default", "path": "cascades/haarcascade_frontalface_default.xml", "enabled": True,
         "min_neighbors": 5, "min_weight": 2},
    ],
    "dnn": {
        "model_path": "models/face_detection_yunet_2026may.onnx",
        "score_threshold": 0.7,
        "nms_threshold": 0.3,
        "top_k": 5000,
    },
    "eyes": {
        "enabled": True,
        "source": "auto",
        "cascade_path": "cascades/haarcascade_eye.xml",
        "scale_factor": 1.1,
        "min_neighbors": 2,
        "min_size": [10, 10],
        "verify_haar_faces": False,
        "min_eyes": 1,
    },
    "detection": {
        "scale_factor": 1.05,
        "min_neighbors": 3,
        "min_size": [24, 24],
        "max_size": None,
        "max_faces": 0,
        "min_cascade_agreement": 2,
        "detect_every_n_frames": 1,
    },
    "tracking": {"enabled": True, "smoothing": 0.4,
                 "face_hold_frames": {"haar": 1, "dnn": 0}, "eye_hold_frames": 15},
    "display": {"enabled": True},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Face and eye detection with a switchable Haar/DNN detector."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to the JSON config file (default: config.json).",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Override the video file path from config.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help="Override the camera index from config.",
    )
    parser.add_argument(
        "--source",
        choices=VALID_VIDEO_SOURCES,
        default=None,
        help=(
            "Override video.source from config. 'auto' tries the configured/"
            "discovered video file, falling back to the camera. 'camera' "
            "always uses the system camera, ignoring any video file."
        ),
    )
    parser.add_argument(
        "--camera",
        action="store_true",
        default=None,
        help="Shorthand for --source camera.",
    )
    parser.add_argument(
        "--detector",
        choices=VALID_DETECTORS,
        default=None,
        help="Override the 'detector' setting from config: 'haar' or 'dnn'.",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=None,
        help=(
            "Override detection.max_faces from config. Only the N largest "
            "faces are kept; 0 keeps every face."
        ),
    )
    parser.add_argument(
        "--cascades",
        default=None,
        help=(
            "Comma-separated cascade names to enable, overriding the "
            "'enabled' flags in config (e.g. 'frontal' or 'frontal,profile')."
        ),
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        default=None,
        help="Do not open a preview window (use on headless/edge devices).",
    )
    parser.add_argument(
        "--save-video",
        default=None,
        metavar="PATH",
        help="Also write the annotated frames to this video file (e.g. out.mp4).",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=None,
        help="Override detectMultiScale scaleFactor from config.",
    )
    parser.add_argument(
        "--min-neighbors",
        type=int,
        default=None,
        help="Override detectMultiScale minNeighbors from config.",
    )
    parser.add_argument(
        "--min-size",
        default=None,
        help="Override detectMultiScale minSize from config, as 'WIDTH,HEIGHT' (e.g. '30,30').",
    )
    parser.add_argument(
        "--max-size",
        default=None,
        help="Override detectMultiScale maxSize from config, as 'WIDTH,HEIGHT'.",
    )
    return parser.parse_args()


def parse_size(value: str, option_name: str) -> tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2:
        log.error("Invalid %s value: %r. Expected 'WIDTH,HEIGHT'.", option_name, value)
        sys.exit(1)
    try:
        return (int(parts[0]), int(parts[1]))
    except ValueError:
        log.error("Invalid %s value: %r. Expected two integers.", option_name, value)
        sys.exit(1)


def load_config(path: str) -> dict:
    if not os.path.isfile(path):
        log.warning("Config file not found: %s. Using built-in defaults.", path)
        return {}

    try:
        with open(path, "r", encoding="utf-8-sig") as f:  # tolerates a Windows BOM
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        log.error("Failed to read config file %s: %s", path, exc)
        sys.exit(1)


def resolve_settings(args: argparse.Namespace) -> dict:
    """Merge the config file over built-in defaults, then apply CLI overrides."""
    user_config = load_config(args.config)

    settings = {
        "video": {**DEFAULT_CONFIG["video"], **user_config.get("video", {})},
        "detector": user_config.get("detector", DEFAULT_CONFIG["detector"]),
        "cascades": user_config.get("cascades", DEFAULT_CONFIG["cascades"]),
        "dnn": {**DEFAULT_CONFIG["dnn"], **user_config.get("dnn", {})},
        "eyes": {**DEFAULT_CONFIG["eyes"], **user_config.get("eyes", {})},
        "detection": {**DEFAULT_CONFIG["detection"], **user_config.get("detection", {})},
        "tracking": {**DEFAULT_CONFIG["tracking"], **user_config.get("tracking", {})},
        "display": {**DEFAULT_CONFIG["display"], **user_config.get("display", {})},
    }

    if args.detector is not None:
        settings["detector"] = args.detector
    if args.max_faces is not None:
        settings["detection"]["max_faces"] = args.max_faces

    if args.video is not None:
        settings["video"]["path"] = args.video
    if args.camera_index is not None:
        settings["video"]["camera_index"] = args.camera_index
    if args.scale_factor is not None:
        settings["detection"]["scale_factor"] = args.scale_factor
    if args.min_neighbors is not None:
        settings["detection"]["min_neighbors"] = args.min_neighbors
    if args.min_size is not None:
        settings["detection"]["min_size"] = list(parse_size(args.min_size, "--min-size"))
    if args.max_size is not None:
        settings["detection"]["max_size"] = list(parse_size(args.max_size, "--max-size"))
    if args.no_display:
        settings["display"]["enabled"] = False
    if args.cascades is not None:
        selected = {name.strip() for name in args.cascades.split(",") if name.strip()}
        for entry in settings["cascades"]:
            entry["enabled"] = entry.get("name") in selected
    if args.source is not None:
        settings["video"]["source"] = args.source
    if args.camera:
        settings["video"]["source"] = "camera"

    if settings["video"].get("source") not in VALID_VIDEO_SOURCES:
        log.warning(
            "Invalid video.source %r (expected one of %s). Using 'auto'.",
            settings["video"].get("source"),
            VALID_VIDEO_SOURCES,
        )
        settings["video"]["source"] = "auto"

    if settings["detector"] not in VALID_DETECTORS:
        log.error(
            "Invalid detector %r (expected one of %s). Fix 'detector' in config.json.",
            settings["detector"],
            VALID_DETECTORS,
        )
        sys.exit(1)

    return settings


def resolve_path(path: str) -> str:
    """Resolve a config path relative to the project root, unless absolute."""
    if not path:
        return path
    return path if os.path.isabs(path) else os.path.join(SCRIPT_DIR, path)


def load_cascade(cascade_path: str) -> cv2.CascadeClassifier:
    if not os.path.isfile(cascade_path):
        log.error("Cascade file not found: %s", cascade_path)
        sys.exit(1)

    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        log.error("Failed to load cascade file (file is invalid or corrupt): %s", cascade_path)
        sys.exit(1)

    log.info("Cascade loaded: %s", cascade_path)
    return cascade


def load_cascades(cascade_entries: list[dict]) -> list[tuple[str, cv2.CascadeClassifier, tuple, dict]]:
    """Load every enabled cascade entry, each paired with a distinct box color.

    The last tuple item holds the entry's optional per-cascade overrides:
    'min_neighbors' and 'min_weight' (minimum detection confidence).
    """
    enabled_entries = [entry for entry in cascade_entries if entry.get("enabled", True)]

    if not enabled_entries:
        log.error(
            "No cascades enabled. Enable at least one in config.json (or via --cascades)."
        )
        sys.exit(1)

    loaded = []
    for i, entry in enumerate(enabled_entries):
        name = entry.get("name", f"cascade_{i}")
        path = resolve_path(entry.get("path", ""))
        cascade = load_cascade(path)
        color = BOX_COLOR_PALETTE[i % len(BOX_COLOR_PALETTE)]
        options = {
            "min_neighbors": entry.get("min_neighbors"),
            "min_weight": float(entry.get("min_weight", 0)),
        }
        loaded.append((name, cascade, color, options))

    return loaded


def load_dnn_detector(dnn_settings: dict):
    """Create OpenCV's YuNet DNN face detector from the configured ONNX model."""
    model_path = resolve_path(dnn_settings.get("model_path", ""))
    if not os.path.isfile(model_path):
        log.error(
            "DNN model not found: %s. Download a face_detection_yunet_*.onnx from "
            "https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet "
            "into models/, or set dnn.model_path in config.json.",
            model_path,
        )
        sys.exit(1)

    try:
        detector = cv2.FaceDetectorYN.create(
            model_path,
            "",
            (320, 320),  # placeholder; the real input size is set from the first frame
            float(dnn_settings["score_threshold"]),
            float(dnn_settings["nms_threshold"]),
            int(dnn_settings["top_k"]),
        )
    except (cv2.error, AttributeError) as exc:
        log.error("Failed to load DNN model %s: %s", model_path, exc)
        sys.exit(1)

    log.info("DNN face detector loaded: %s", model_path)
    return detector


def box_iou(a: tuple, b: tuple) -> float:
    """Intersection-over-union of two (x, y, w, h, ...) boxes."""
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter) if inter else 0.0


def detect_faces_haar(gray, cascades: list, detect_kwargs: dict, min_agreement: int = 1) -> list[tuple]:
    """Find faces with the first cascade and have the others confirm them.

    A single Haar cascade fires on clothing and other texture, but different
    cascades rarely make the same mistake. The first cascade scans the whole
    frame and proposes candidate faces; each other cascade then re-checks only
    a small region around each candidate (cheap) and votes if it finds an
    overlapping box (IoU > MERGE_IOU). A candidate is kept when it has at least
    min_agreement votes, its own included. A cascade only votes for a box whose
    confidence reaches its 'min_weight'. Returns (x, y, w, h, color, None).
    """
    _, primary, color, primary_options = cascades[0]
    rects, _, weights = primary.detectMultiScale3(
        gray, outputRejectLevels=True, **cascade_kwargs(detect_kwargs, primary_options)
    )

    faces = []
    for rect, weight in zip(rects, np.ravel(weights)):
        if weight < primary_options["min_weight"]:
            continue
        box = tuple(int(v) for v in rect)
        votes = 1
        for _, cascade, _, options in cascades[1:]:
            if votes >= min_agreement:
                break
            if cascade_confirms(gray, cascade, box, detect_kwargs, options):
                votes += 1
        if votes >= min_agreement:
            faces.append((*box, color, None))
    return drop_boxes_below_heads(faces)


def cascade_kwargs(detect_kwargs: dict, options: dict) -> dict:
    """detectMultiScale kwargs for one cascade, applying its own min_neighbors."""
    kwargs = dict(detect_kwargs)
    if options["min_neighbors"] is not None:
        kwargs["minNeighbors"] = options["min_neighbors"]
    return kwargs


def cascade_confirms(gray, cascade, box: tuple, detect_kwargs: dict, options: dict) -> bool:
    """True if the cascade finds a face overlapping box, searching only around it."""
    x, y, w, h = box
    frame_h, frame_w = gray.shape[:2]
    x0, y0 = max(x - int(w * 0.3), 0), max(y - int(h * 0.3), 0)
    x1, y1 = min(x + w + int(w * 0.3), frame_w), min(y + h + int(h * 0.3), frame_h)

    kwargs = cascade_kwargs(detect_kwargs, options)
    kwargs.pop("maxSize", None)
    min_side = max(int(w * 0.6), 8)
    kwargs["minSize"] = (min_side, min_side)

    rects, _, weights = cascade.detectMultiScale3(
        gray[y0:y1, x0:x1], outputRejectLevels=True, **kwargs
    )
    for (rx, ry, rw, rh), weight in zip(rects, np.ravel(weights)):
        if weight >= options["min_weight"] and box_iou((rx + x0, ry + y0, rw, rh), box) > MERGE_IOU:
            return True
    return False


def drop_boxes_below_heads(faces: list[tuple]) -> list[tuple]:
    """Drop boxes that sit directly under another face box (torso/clothing hits).

    Heads are above bodies, so a box lying below a face box, horizontally
    aligned with it and not hugely larger (a much larger box is a different,
    nearer person), is almost always the same person's body, not a face.
    """
    kept = []
    for b in faces:
        b_cx = b[0] + b[2] / 2
        is_body = False
        for a in faces:
            if a is b:
                continue
            a_cx = a[0] + a[2] / 2
            below = a[1] + 0.8 * a[3] <= b[1] <= a[1] + 4 * a[3]
            aligned = abs(a_cx - b_cx) < STACK_ALIGN * max(a[2], b[2])
            similar_scale = b[2] <= 3 * a[2]
            if below and aligned and similar_scale:
                is_body = True
                break
        if not is_body:
            kept.append(b)
    return kept


def detect_faces_dnn(frame, detector) -> list[tuple]:
    """Run the DNN detector; returns (x, y, w, h, color, eye_points) per detection.

    eye_points are the two eye-center landmarks YuNet reports for each face.
    """
    frame_h, frame_w = frame.shape[:2]
    detector.setInputSize((frame_w, frame_h))
    _, detections = detector.detect(frame)
    if detections is None:
        return []

    faces = []
    for det in detections:
        # Row layout: x, y, w, h, 10 landmark coords, score. Clip to the frame.
        x0, y0 = max(int(det[0]), 0), max(int(det[1]), 0)
        x1, y1 = min(int(det[0] + det[2]), frame_w), min(int(det[1] + det[3]), frame_h)
        if x1 > x0 and y1 > y0:
            eye_points = ((int(det[4]), int(det[5])), (int(det[6]), int(det[7])))
            faces.append((x0, y0, x1 - x0, y1 - y0, DNN_BOX_COLOR, eye_points))
    return faces


def eyes_from_landmarks(face: tuple) -> list[tuple]:
    """Turn a DNN face's eye landmarks into [left, right] (x, y, w, h) eye boxes."""
    side = max(int(face[2] * 0.22), 6)
    points = sorted(face[5])  # left-most point first
    return [(px - side // 2, py - side // 2, side, side) for px, py in points]


def detect_eyes(gray, face: tuple, eye_cascade: cv2.CascadeClassifier, eye_settings: dict) -> list:
    """Find the eyes inside a face box; returns [left, right], None where not found.

    Eyes sit in the upper half of a face, so only that band is searched;
    this rejects nostril/mouth false positives. The band is upscaled and
    contrast-enhanced first: the eye cascade barely fires on low-contrast
    (hazy/dim) footage otherwise. Candidates must also sit where eyes
    anatomically are (EYE_ZONES), which rejects eyebrows, glasses frames and
    hair. Returned boxes are absolute (x, y, w, h).
    """
    x, y, w, h = face[:4]
    min_w, min_h = eye_settings["min_size"]
    band_top = y + int(h * 0.15)
    band_bottom = y + int(h * 0.60)
    max_size = (w // 2, h // 2)
    if w < 2 * min_w or band_bottom - band_top < min_h or max_size[1] < min_h:
        return [None, None]

    scale = EYE_UPSCALE
    roi = cv2.resize(
        gray[band_top:band_bottom, x:x + w], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
    )
    roi = EYE_CLAHE.apply(roi)
    candidates = eye_cascade.detectMultiScale(
        roi,
        scaleFactor=eye_settings["scale_factor"],
        minNeighbors=eye_settings["min_neighbors"],
        minSize=(min_w * scale, min_h * scale),
        maxSize=(max_size[0] * scale, max_size[1] * scale),
    )

    # Keep the largest plausible candidate on each side of the face midline.
    best = [None, None]
    for (ex, ey, ew, eh) in candidates:
        eye = (x + int(ex / scale), band_top + int(ey / scale), int(ew / scale), int(eh / scale))
        # Center and size relative to the face box.
        rel_x = (eye[0] + eye[2] / 2 - x) / w
        rel_y = (eye[1] + eye[3] / 2 - y) / h
        rel_size = eye[2] / w
        side = 0 if rel_x < 0.5 else 1
        x_lo, x_hi = EYE_ZONE_X[side]
        if not (x_lo <= rel_x <= x_hi and EYE_ZONE_Y[0] <= rel_y <= EYE_ZONE_Y[1]
                and EYE_SIZE[0] <= rel_size <= EYE_SIZE[1]):
            continue
        if best[side] is None or eye[2] * eye[3] > best[side][2] * best[side][3]:
            best[side] = eye

    # Two boxes on top of each other are one eye (typical on turned faces): keep the larger.
    left, right = best
    if left is not None and right is not None:
        separation = abs((left[0] + left[2] / 2) - (right[0] + right[2] / 2))
        if separation < EYE_MIN_SEPARATION * w:
            if left[2] * left[3] >= right[2] * right[3]:
                right = None
            else:
                left = None
    return [left, right]


def track_affinity(track_box: list, det_box: tuple) -> float:
    """How well a detection continues a track; > 0 means they match.

    Overlapping boxes score their IoU. Small or fast faces can move further
    than their own width between detection passes and stop overlapping, so
    boxes whose centers are within 0.75 face widths still match, with a lower
    score than any real overlap.
    """
    overlap = box_iou(track_box, det_box)
    if overlap > MERGE_IOU:
        return overlap
    dx = (track_box[0] + track_box[2] / 2) - (det_box[0] + det_box[2] / 2)
    dy = (track_box[1] + track_box[3] / 2) - (det_box[1] + det_box[3] / 2)
    limit = 0.75 * max(track_box[2], det_box[2])
    distance = (dx * dx + dy * dy) ** 0.5
    return MERGE_IOU * (1 - distance / limit) if distance < limit else 0.0


class FaceTrack:
    """One tracked face: smoothed box plus smoothed, briefly held eye boxes."""

    # Eyes are stored relative to the face box (center x, center y, width,
    # height as fractions of the face box), so they keep their place on the
    # face when it moves or grows (a person walking toward the camera).

    def __init__(self, face: tuple, eyes: list):
        self.box = [float(v) for v in face[:4]]
        self.color = face[4]
        self.eyes = [None, None]
        self.face_missed = 0
        self.eye_missed = [0, 0]
        self.observe(face, eyes, 1.0, 0)

    @staticmethod
    def _to_relative(eye: tuple, face: tuple) -> list:
        fx, fy, fw, fh = face[:4]
        ex, ey, ew, eh = eye
        return [(ex + ew / 2 - fx) / fw, (ey + eh / 2 - fy) / fh, ew / fw, eh / fh]

    def observe(self, face: tuple, eyes: list, alpha: float, eye_hold: int) -> None:
        """Blend a new detection in. alpha is the weight given to the new values."""
        self.box = [o + alpha * (n - o) for o, n in zip(self.box, face[:4])]
        self.color = face[4]
        self.face_missed = 0

        eyes = list(eyes) + [None] * (2 - len(eyes))
        for side in (0, 1):
            new, cur = eyes[side], self.eyes[side]
            if new is not None:
                rel = self._to_relative(new, face)
                self.eyes[side] = rel if cur is None else [o + alpha * (n - o) for o, n in zip(cur, rel)]
                self.eye_missed[side] = 0
            elif cur is not None:
                self.eye_missed[side] += 1
                if self.eye_missed[side] > eye_hold:
                    self.eyes[side] = None

        # A held eye that ends up on top of the other eye is stale: drop it.
        left, right = self.eyes
        if left is not None and right is not None and abs(left[0] - right[0]) < EYE_MIN_SEPARATION:
            self.eyes[0 if self.eye_missed[0] >= self.eye_missed[1] else 1] = None

    def snapshot(self) -> tuple:
        """(face, eyes) in the same shape the detectors produce, ints for drawing."""
        fx, fy, fw, fh = self.box
        face = (int(round(fx)), int(round(fy)), int(round(fw)), int(round(fh)), self.color, None)
        eyes = []
        for rel in self.eyes:
            if rel is None:
                eyes.append(None)
                continue
            rcx, rcy, rw, rh = rel
            ew, eh = rw * fw, rh * fh
            eyes.append((int(round(fx + rcx * fw - ew / 2)), int(round(fy + rcy * fh - eh / 2)),
                         int(round(ew)), int(round(eh))))
        return face, eyes


class FaceTracker:
    """Matches detections across frames to remove box and eye flicker.

    Detectors miss a face or an eye on some frames and jitter on others. The
    tracker smooths positions (exponential moving average), keeps a face
    for face_hold missed detection passes, and keeps each eye for eye_hold.
    """

    def __init__(self, smoothing: float, face_hold: int, eye_hold: int):
        self.alpha = min(max(smoothing, 0.05), 1.0)
        self.face_hold = face_hold
        self.eye_hold = eye_hold
        self.tracks: list[FaceTrack] = []

    def update(self, detections: list) -> None:
        """detections: (face, eyes) pairs from one detection pass."""
        pairs = sorted(
            (
                (track_affinity(track.box, det[0]), ti, di)
                for ti, track in enumerate(self.tracks)
                for di, det in enumerate(detections)
            ),
            reverse=True,
        )
        used_tracks, used_dets = set(), set()
        for affinity, ti, di in pairs:
            if affinity <= 0 or ti in used_tracks or di in used_dets:
                continue
            used_tracks.add(ti)
            used_dets.add(di)
            face, eyes = detections[di]
            self.tracks[ti].observe(face, eyes, self.alpha, self.eye_hold)

        for ti, track in enumerate(self.tracks):
            if ti not in used_tracks:
                track.face_missed += 1
        self.tracks = [t for t in self.tracks if t.face_missed <= self.face_hold]
        self.tracks += [FaceTrack(*det) for di, det in enumerate(detections) if di not in used_dets]

    def current(self) -> list:
        return [track.snapshot() for track in self.tracks]


def list_videos(videos_dir: str) -> list[str]:
    """Return the video filenames found directly in videos_dir, sorted."""
    if not os.path.isdir(videos_dir):
        return []
    return sorted(
        entry for entry in os.listdir(videos_dir) if entry.lower().endswith(VIDEO_EXTENSIONS)
    )


def discover_video(videos_dir: str) -> str | None:
    """Return the first video file found in videos_dir (alphabetically), or None."""
    videos = list_videos(videos_dir)
    if not videos:
        return None

    if len(videos) > 1:
        log.info(
            "Multiple video files found in videos/: %s. Set video.path in config.json "
            "(or --video) to pick one explicitly; using '%s' (first alphabetically).",
            ", ".join(videos),
            videos[0],
        )

    return os.path.join(videos_dir, videos[0])


def resolve_video_path(path: str) -> str:
    """Resolve a configured video path.

    Tries, in order: as given if absolute, relative to the project root
    (e.g. "videos/clip.mp4"), then as a bare filename inside videos/
    (e.g. "clip.mp4"). Falls back to the project-root-relative form if
    none of those exist, so the caller's "not found" warning still shows
    a sensible path.
    """
    if not path or os.path.isabs(path):
        return path

    project_relative = os.path.join(SCRIPT_DIR, path)
    if os.path.isfile(project_relative):
        return project_relative

    videos_relative = os.path.join(VIDEOS_DIR, path)
    if os.path.isfile(videos_relative):
        return videos_relative

    return project_relative


def open_camera(camera_index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        log.error("Could not open system camera at index %d.", camera_index)
        sys.exit(1)

    log.info("System camera opened at index %d.", camera_index)
    return cap


def open_capture(video_path: str | None, camera_index: int) -> cv2.VideoCapture:
    """Open a video source, falling back to the system camera.

    Resolution order: the configured/--video path, then the first video
    file found in videos/, then the system camera. Falls through to the
    next option whenever the current one is missing or OpenCV cannot open
    it (e.g. unsupported codec).
    """
    if not video_path:
        video_path = discover_video(VIDEOS_DIR)
        if video_path:
            log.info("No video path configured; using discovered file: %s", video_path)

    if video_path:
        if not os.path.isfile(video_path):
            available = list_videos(VIDEOS_DIR)
            hint = f" Available in videos/: {', '.join(available)}." if available else ""
            log.warning(
                "Video path not found: %s. Falling back to system camera.%s", video_path, hint
            )
        else:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                log.info("Video file loaded: %s", video_path)
                return cap
            log.warning(
                "Could not open video file (unsupported/corrupt): %s. Falling back to system camera.",
                video_path,
            )
            cap.release()

    return open_camera(camera_index)


class Pipeline:
    """Face detection, eye detection and tracking: everything main() does per frame.

    process(frame) returns the (face, eyes) pairs to draw. The time each
    stage took for the latest frame is left in self.timings (milliseconds):
    'detect_ms' (grayscale + face detection), 'eyes_ms' and 'track_ms'.
    """

    def __init__(self, settings: dict):
        self.detector_type = settings["detector"]
        log.info("Face detector: %s", self.detector_type)

        self.cascades = []
        self.dnn_detector = None
        if self.detector_type == "dnn":
            self.dnn_detector = load_dnn_detector(settings["dnn"])
        else:
            self.cascades = load_cascades(settings["cascades"])
            for name, _, color, _ in self.cascades:
                log.info("Cascade '%s' enabled (box color BGR=%s).", name, color)

        self.eye_settings = settings["eyes"]
        self.eyes_enabled = self.eye_settings.get("enabled", True)
        self.eye_source = self.eye_settings.get("source", "auto")
        if self.eye_source not in VALID_EYE_SOURCES:
            log.error("Invalid eyes.source %r (expected one of %s).", self.eye_source, VALID_EYE_SOURCES)
            sys.exit(1)
        if self.eye_source == "auto":
            self.eye_source = "landmarks" if self.detector_type == "dnn" else "cascade"
        elif self.eye_source == "landmarks" and self.detector_type != "dnn":
            log.warning("eyes.source 'landmarks' needs the dnn detector; using the eye cascade.")
            self.eye_source = "cascade"

        self.eye_cascade = None
        if self.eyes_enabled and (
            self.eye_source == "cascade" or self.eye_settings.get("verify_haar_faces")
        ):
            self.eye_cascade = load_cascade(resolve_path(self.eye_settings["cascade_path"]))
        self.verify_with_eyes = (
            self.detector_type == "haar"
            and self.eye_cascade is not None
            and self.eye_settings.get("verify_haar_faces", False)
        )
        self.min_eyes = self.eye_settings.get("min_eyes", 1)
        self.max_faces = settings["detection"].get("max_faces") or 0
        self.min_agreement = max(int(settings["detection"].get("min_cascade_agreement") or 1), 1)
        if self.cascades and self.min_agreement > len(self.cascades):
            log.warning(
                "detection.min_cascade_agreement is %d but only %d cascade(s) are enabled; "
                "using %d.", self.min_agreement, len(self.cascades), len(self.cascades),
            )
            self.min_agreement = len(self.cascades)
        if self.cascades:
            log.info("A face needs %d of %d cascade(s) to agree.", self.min_agreement, len(self.cascades))
        if self.verify_with_eyes:
            log.info("Haar faces must contain at least %d eye(s) to be kept.", self.min_eyes)

        detection = settings["detection"]
        self.detect_kwargs = {
            "scaleFactor": detection["scale_factor"],
            "minNeighbors": detection["min_neighbors"],
        }
        if detection.get("min_size"):
            self.detect_kwargs["minSize"] = tuple(detection["min_size"])
        if detection.get("max_size"):
            self.detect_kwargs["maxSize"] = tuple(detection["max_size"])

        self.detect_every = max(int(detection.get("detect_every_n_frames") or 1), 1)
        tracking = settings["tracking"]
        self.tracker = None
        if tracking.get("enabled", True):
            # face_hold_frames may be one number or {"haar": n, "dnn": n}: the DNN
            # detector is steady, so holding a lost face only leaves ghost boxes.
            face_hold = tracking["face_hold_frames"]
            if isinstance(face_hold, dict):
                face_hold = face_hold.get(self.detector_type, 1)
            self.tracker = FaceTracker(
                float(tracking["smoothing"]),
                int(face_hold),
                int(tracking["eye_hold_frames"]),
            )

        self.frame_count = 0
        self.results = []  # (face, eyes) pairs from the latest detection pass
        self.timings = {"detect_ms": 0.0, "eyes_ms": 0.0, "track_ms": 0.0}

    def _eyes_for(self, gray, face: tuple) -> list:
        if not self.eyes_enabled:
            return []
        if self.eye_source == "landmarks":
            return eyes_from_landmarks(face)
        if self.eye_cascade is not None:
            return detect_eyes(gray, face, self.eye_cascade, self.eye_settings)
        return []

    def process(self, frame) -> list:
        now = time.perf_counter
        detect_s = eyes_s = track_s = 0.0

        # Detection is the expensive part; on skipped frames the previous
        # frame's boxes are redrawn.
        if self.frame_count % self.detect_every == 0:
            start = now()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.dnn_detector is not None:
                faces = detect_faces_dnn(frame, self.dnn_detector)
            else:
                faces = detect_faces_haar(gray, self.cascades, self.detect_kwargs, self.min_agreement)
            # Largest faces first, so max_faces keeps the face closest to the camera.
            faces.sort(key=lambda f: f[2] * f[3], reverse=True)
            detect_s = now() - start

            self.results = []
            for face in faces:
                start = now()
                eyes = self._eyes_for(gray, face)
                eyes_s += now() - start
                if self.verify_with_eyes and sum(e is not None for e in eyes) < self.min_eyes:
                    continue

                self.results.append((face, eyes))
                if self.max_faces and len(self.results) >= self.max_faces:
                    break

            if self.tracker is not None:
                start = now()
                self.tracker.update(self.results)
                track_s += now() - start

        start = now()
        output = self.tracker.current() if self.tracker is not None else self.results
        track_s += now() - start

        self.frame_count += 1
        self.timings = {
            "detect_ms": detect_s * 1000,
            "eyes_ms": eyes_s * 1000,
            "track_ms": track_s * 1000,
        }
        return output


def main() -> None:
    args = parse_args()
    settings = resolve_settings(args)
    pipeline = Pipeline(settings)

    camera_index = settings["video"].get("camera_index", 0)
    if settings["video"].get("source") == "camera":
        log.info("video.source is 'camera'; using the system camera and ignoring any video file.")
        cap = open_camera(camera_index)
    else:
        video_path = resolve_video_path(settings["video"].get("path") or "")
        cap = open_capture(video_path or None, camera_index)

    display_enabled = settings["display"]["enabled"]
    frame_count = 0
    writer = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log.info("No more frames to read. Stopping.")
                break

            for face, eyes in pipeline.process(frame):
                x, y, w, h, color = face[:5]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                for eye in eyes:
                    if eye is not None:
                        ex, ey, ew, eh = eye
                        cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), EYE_BOX_COLOR, 1)

            frame_count += 1

            if args.save_video:
                if writer is None:
                    frame_h, frame_w = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        args.save_video,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        cap.get(cv2.CAP_PROP_FPS) or 25.0,
                        (frame_w, frame_h),
                    )
                writer.write(frame)

            if display_enabled:
                try:
                    cv2.imshow("Face Detection", frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        log.info("Quit key pressed. Stopping.")
                        break
                except cv2.error as exc:
                    log.warning("Display unavailable (%s). Continuing headless.", exc)
                    display_enabled = False

        log.info("Finished processing. Total frames processed: %d", frame_count)

    except KeyboardInterrupt:
        log.info("Interrupted by user. Stopping.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if display_enabled:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
