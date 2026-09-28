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
EYE_BOX_COLOR = (0, 0, 255)  # red, so eyes stand out from any face box color
DNN_BOX_COLOR = BOX_COLOR_PALETTE[0]

VALID_VIDEO_SOURCES = ("auto", "camera")
VALID_DETECTORS = ("haar", "dnn")

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
        "cascade_path": "cascades/haarcascade_eye.xml",
        "scale_factor": 1.1,
        "min_neighbors": 5,
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
    },
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
        with open(path, "r", encoding="utf-8") as f:
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
    """Run every enabled cascade and merge their detections into faces.

    A single Haar cascade fires on clothing and other texture, but different
    cascades rarely make the same mistake. Boxes from different cascades that
    overlap (IoU > MERGE_IOU) count as one face, keeping the earlier cascade's
    box; a face is kept only if at least min_agreement cascades found it. A
    cascade only votes for a box whose confidence reaches its 'min_weight'.
    Returns (x, y, w, h, color) per face.
    """
    merged = []  # [x, y, w, h, color, set of voting cascade indexes]
    for idx, (_, cascade, color, options) in enumerate(cascades):
        kwargs = dict(detect_kwargs)
        if options["min_neighbors"] is not None:
            kwargs["minNeighbors"] = options["min_neighbors"]
        rects, _, weights = cascade.detectMultiScale3(gray, outputRejectLevels=True, **kwargs)
        for rect, weight in zip(rects, np.ravel(weights)):
            if weight < options["min_weight"]:
                continue
            box = tuple(int(v) for v in rect)
            for face in merged:
                if idx not in face[5] and box_iou(box, face) > MERGE_IOU:
                    face[5].add(idx)
                    break
            else:
                merged.append([*box, color, {idx}])

    return [tuple(face[:5]) for face in merged if len(face[5]) >= min_agreement]


def detect_faces_dnn(frame, detector) -> list[tuple]:
    """Run the DNN detector; returns (x, y, w, h, color) per detection."""
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
            faces.append((x0, y0, x1 - x0, y1 - y0, DNN_BOX_COLOR))
    return faces


def detect_eyes(gray, face: tuple, eye_cascade: cv2.CascadeClassifier, eye_settings: dict) -> list[tuple]:
    """Find up to two eyes (one per side) inside a face box.

    Eyes sit in the upper half of a face, so only that band is searched;
    this rejects nostril/mouth false positives. Returns absolute (x, y, w, h).
    """
    x, y, w, h = face[:4]
    min_w, min_h = eye_settings["min_size"]
    band_top = y + int(h * 0.15)
    band_bottom = y + int(h * 0.60)
    max_size = (w // 2, h // 2)
    if w < 2 * min_w or band_bottom - band_top < min_h or max_size[1] < min_h:
        return []

    roi = gray[band_top:band_bottom, x:x + w]
    candidates = eye_cascade.detectMultiScale(
        roi,
        scaleFactor=eye_settings["scale_factor"],
        minNeighbors=eye_settings["min_neighbors"],
        minSize=(min_w, min_h),
        maxSize=max_size,
    )

    # Keep the largest candidate on each side of the face midline.
    left = right = None
    for (ex, ey, ew, eh) in candidates:
        eye = (x + int(ex), band_top + int(ey), int(ew), int(eh))
        if ex + ew / 2 < w / 2:
            if left is None or ew * eh > left[2] * left[3]:
                left = eye
        elif right is None or ew * eh > right[2] * right[3]:
            right = eye
    return [eye for eye in (left, right) if eye is not None]


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


def main() -> None:
    args = parse_args()
    settings = resolve_settings(args)

    detector_type = settings["detector"]
    log.info("Face detector: %s", detector_type)

    cascades = []
    dnn_detector = None
    if detector_type == "dnn":
        dnn_detector = load_dnn_detector(settings["dnn"])
    else:
        cascades = load_cascades(settings["cascades"])
        for name, _, color, _ in cascades:
            log.info("Cascade '%s' enabled (box color BGR=%s).", name, color)

    eye_settings = settings["eyes"]
    eye_cascade = None
    if eye_settings.get("enabled", True):
        eye_cascade = load_cascade(resolve_path(eye_settings["cascade_path"]))
    verify_with_eyes = (
        detector_type == "haar"
        and eye_cascade is not None
        and eye_settings.get("verify_haar_faces", True)
    )
    min_eyes = eye_settings.get("min_eyes", 1)
    max_faces = settings["detection"].get("max_faces") or 0
    min_agreement = max(int(settings["detection"].get("min_cascade_agreement") or 1), 1)
    if cascades and min_agreement > len(cascades):
        log.warning(
            "detection.min_cascade_agreement is %d but only %d cascade(s) are enabled; "
            "using %d.", min_agreement, len(cascades), len(cascades),
        )
        min_agreement = len(cascades)
    if cascades:
        log.info("A face needs %d of %d cascade(s) to agree.", min_agreement, len(cascades))
    if verify_with_eyes:
        log.info("Haar faces must contain at least %d eye(s) to be kept.", min_eyes)

    camera_index = settings["video"].get("camera_index", 0)
    if settings["video"].get("source") == "camera":
        log.info("video.source is 'camera'; using the system camera and ignoring any video file.")
        cap = open_camera(camera_index)
    else:
        video_path = resolve_video_path(settings["video"].get("path") or "")
        cap = open_capture(video_path or None, camera_index)

    scale_factor = settings["detection"]["scale_factor"]
    min_neighbors = settings["detection"]["min_neighbors"]
    min_size = settings["detection"].get("min_size")
    max_size = settings["detection"].get("max_size")
    detect_kwargs = {
        "scaleFactor": scale_factor,
        "minNeighbors": min_neighbors,
    }
    if min_size:
        detect_kwargs["minSize"] = tuple(min_size)
    if max_size:
        detect_kwargs["maxSize"] = tuple(max_size)

    display_enabled = settings["display"]["enabled"]
    frame_count = 0
    writer = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log.info("No more frames to read. Stopping.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if dnn_detector is not None:
                faces = detect_faces_dnn(frame, dnn_detector)
            else:
                faces = detect_faces_haar(gray, cascades, detect_kwargs, min_agreement)

            # Largest faces first, so max_faces keeps the face closest to the camera.
            faces.sort(key=lambda f: f[2] * f[3], reverse=True)
            kept = 0
            for face in faces:
                eyes = detect_eyes(gray, face, eye_cascade, eye_settings) if eye_cascade is not None else []
                if verify_with_eyes and len(eyes) < min_eyes:
                    continue

                x, y, w, h, color = face
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), EYE_BOX_COLOR, 1)

                kept += 1
                if max_faces and kept >= max_faces:
                    break

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
