# Face Detection

Real-time face and eye detection with a switchable detector: OpenCV Haar
cascades or OpenCV's DNN face detector (YuNet). The detector, which
cascades run, and which video source is used are controlled by
[`config.json`](config.json), so switching between detectors, cascades or
footage is a config edit, not a code change. Reads frames from a configured video
file when one is available, and automatically falls back to the system
camera otherwise. Built to run unmodified on a desktop for development
and inside a container on an edge device for deployment.

## Project structure

```
Face_Detection/
├── app.py                          # Main application
├── config.json                     # Selects active cascade(s), video source, detection params
├── cascades/                       # Pre-trained Haar cascade models
│   ├── haarcascade_frontalface_alt.xml
│   ├── haarcascade_frontalface_default.xml
│   ├── haarcascade_profileface.xml
│   └── haarcascade_eye.xml
├── models/                         # DNN face detector model
│   └── face_detection_yunet_2026may.onnx
├── requirements.txt                # Dependencies for local/dev use (with GUI display)
├── requirements-docker.txt         # Dependencies for the container (headless)
├── Dockerfile                      # Container build for edge deployment
├── .dockerignore
├── .gitignore
├── videos/                         # Local video files (git-ignored, see below)
│   └── .gitkeep
└── .venv/                          # Local virtual environment (git-ignored)
```

## Configuration (`config.json`)

```json
{
  "video": {
    "path": "",
    "camera_index": 0,
    "source": "auto"
  },
  "detector": "dnn",
  "cascades": [
    { "name": "frontal_alt", "path": "cascades/haarcascade_frontalface_alt.xml", "enabled": true },
    { "name": "frontal_default", "path": "cascades/haarcascade_frontalface_default.xml", "enabled": true }
  ],
  "dnn": {
    "model_path": "models/face_detection_yunet_2026may.onnx",
    "score_threshold": 0.7,
    "nms_threshold": 0.3,
    "top_k": 5000
  },
  "eyes": {
    "enabled": true,
    "cascade_path": "cascades/haarcascade_eye.xml",
    "scale_factor": 1.05,
    "min_neighbors": 3,
    "min_size": [10, 10],
    "verify_haar_faces": true,
    "min_eyes": 1
  },
  "detection": {
    "scale_factor": 1.1,
    "min_neighbors": 8,
    "min_size": [60, 60],
    "max_size": null,
    "max_faces": 1
  },
  "display": {
    "enabled": true
  }
}
```

- **`video.path`** — path to a video file, relative to the project root
  (or absolute), or a bare filename resolved inside `videos/`. Leave it
  `""` to auto-discover the first video file in `videos/`.
- **`video.camera_index`** — system camera index used when no video file
  is configured or found, or when `video.source` is `"camera"`.
- **`video.source`** — `"auto"` (default) tries the configured/discovered
  video file first, falling back to the camera if it's missing/unreadable.
  `"camera"` always uses the system camera and skips video resolution
  entirely, even if video files exist in `videos/`. This is the switch to
  use when you want to test live camera input without moving your video
  files out of the way. Override per-run with `--source camera` or the
  `--camera` shorthand, without editing the file.
- **`detector`** — `"haar"` or `"dnn"`. This is the toggle between the
  Haar cascade(s) listed in `cascades` and OpenCV's DNN face detector
  (YuNet, configured in `dnn`). Override per-run with `--detector dnn`.
  The DNN detector is far less prone to boxes on bodies/clothing.
- **`dnn`** — used when `detector` is `"dnn"`. `model_path` is the YuNet
  ONNX file; `score_threshold` (0–1) is the minimum confidence to keep a
  face (raise it for fewer false positives); `nms_threshold` and `top_k`
  control overlap suppression.
- **`eyes`** — eye detection with `haarcascade_eye.xml`, run inside every
  kept face box (upper part of the face only, at most one eye per side)
  and drawn as red boxes. Set `enabled: false` to turn it off. With the
  Haar detector, `verify_haar_faces: true` also discards any face box
  containing fewer than `min_eyes` eyes — this is what removes Haar boxes
  that land on a torso or background. Eye verification is not applied to
  the DNN detector, which is reliable on its own.
- **`detection.max_faces`** — keep only the N largest faces per frame
  (`1` = the face closest to the camera). `0` keeps every face. Override
  with `--max-faces`. Note this selects by size, not identity: it doesn't
  recognise *whose* face it is.
- **`cascades`** — (Haar only) an ordered list of cascades. Each entry has a `name`
  (used in logs and for `--cascades` overrides), a `path`, and an
  `enabled` flag. Set `enabled: false` to skip a cascade without deleting
  it from the file, or add a new entry to run a custom cascade you drop
  into `cascades/` — no code changes required. Each enabled cascade draws
  its detections in a different box color, cycled in list order.
  At least one cascade must be enabled.
- **`detection.scale_factor` / `detection.min_neighbors`** —
  `detectMultiScale` parameters, applied to every enabled cascade
  (Haar only).
- **`detection.min_size` / `detection.max_size`** — (Haar only) `[width, height]` in
  pixels, or `null` to leave unconstrained. Bounds the face sizes
  `detectMultiScale` will consider; narrowing this range to match the
  actual face sizes in your footage reduces both false positives and
  stray/misaligned boxes. See the troubleshooting note below.
- **`display.enabled`** — whether to open a preview window.

Use `--config <path>` to point `app.py` at an alternate config file (e.g.
`config.edge.json` with a lighter cascade set for a resource-constrained
device), instead of editing `config.json` in place.

## How it works

1. Loads `config.json` (or the file given via `--config`), merges it over
   built-in defaults, then applies any CLI overrides.
2. Loads every cascade entry with `enabled: true`.
3. Opens a video source:
   - If `video.source` is `"camera"`, opens the system camera
     (`camera_index`) directly — no video file is looked at.
   - Otherwise (`"auto"`), in order: the configured/`--video` path if it
     exists and can be opened, then the first video file found in the
     `videos/` directory, then the system camera. Each step falls through
     to the next if the file is missing or OpenCV can't open it, with a
     warning logged explaining why.
4. For every frame: detects faces with the configured detector (Haar
   `detectMultiScale` per enabled cascade, or the DNN detector), sorts
   them largest first, runs eye detection inside each face, drops Haar
   faces without eyes, keeps at most `max_faces`, and draws the face box
   (cascade color / green for DNN) plus red eye boxes.
5. Displays the annotated frame in a window, unless display is disabled or
   unavailable (e.g. inside a headless container), in which case it keeps
   processing without a preview.
6. Releases the capture and closes any window on exit, on end of stream,
   on `q` / `Esc`, or on `Ctrl+C`.

All major failure points (missing config, missing/corrupt cascade file,
unreadable video, unavailable camera, display errors) are caught and
logged with a clear message instead of crashing with a raw traceback.

> `haarcascade_profileface.xml` is trained on left-facing profiles only,
> so right-facing profiles in a video/feed may not be detected. This is a
> limitation of the stock OpenCV cascade, not of this app.

## Local setup (Windows, PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

## Usage

```powershell
# Uses config.json as-is: both frontal cascades enabled, video auto-discovered from videos/
.venv\Scripts\python app.py

# Run with only one cascade for this run, without editing config.json
.venv\Scripts\python app.py --cascades frontal_default

# Point at a different config file entirely (e.g. a lighter edge profile)
.venv\Scripts\python app.py --config config.edge.json

# Override just the video file for this run
.venv\Scripts\python app.py --video videos\other-clip.mp4

# Override the camera index (used when no video is found/given)
.venv\Scripts\python app.py --camera-index 1

# Force the system camera for this run, even with video files present in videos/
.venv\Scripts\python app.py --camera
# equivalent: .venv\Scripts\python app.py --source camera

# Headless run (no preview window) — used for edge/servers
.venv\Scripts\python app.py --no-display
```

CLI flags override the corresponding config value for that run only;
`config.json` itself is never modified.

| Argument            | Default            | Description                                   |
|---------------------|---------------------|------------------------------------------------|
| `--config`          | `config.json`        | Path to the JSON config file to use.          |
| `--video`           | *(from config)*      | Override the video file path from config.     |
| `--camera-index`    | *(from config)*      | Override the camera index from config.        |
| `--source`          | *(from config)*      | Override `video.source`: `auto` or `camera`.  |
| `--camera`          | off                   | Shorthand for `--source camera`.              |
| `--detector`        | *(from config)*      | Override `detector`: `haar` or `dnn`.         |
| `--max-faces`       | *(from config)*      | Keep only the N largest faces (`0` = all).    |
| `--cascades`        | *(from config)*      | Comma-separated cascade `name`s to enable, overriding config's `enabled` flags (e.g. `frontal_default` or `frontal_alt,frontal_default`). |
| `--no-display`      | *(from config)*      | Force-disables the preview window.            |
| `--scale-factor`    | *(from config)*      | Override `detectMultiScale` scaleFactor.      |
| `--min-neighbors`   | *(from config)*      | Override `detectMultiScale` minNeighbors.     |
| `--min-size`        | *(from config)*      | Override `detectMultiScale` minSize, as `WIDTH,HEIGHT` (e.g. `30,30`). |
| `--max-size`        | *(from config)*      | Override `detectMultiScale` maxSize, as `WIDTH,HEIGHT`. |

## Troubleshooting: missed or misaligned faces with 3+ people

Haar cascades score detections by clustering overlapping matches from a
multi-scale scan; a face that's smaller, angled, or partially occluded by
another person in the frame gets fewer of these overlapping matches, so
it's more likely to be missed or grouped into a loose/offset box. This
gets more likely as more people enter the frame. Things to try, in order
of effort:

1. **Constrain `min_size`/`max_size`** to the actual pixel range faces
   occupy in your footage (larger for close-up shots, smaller for a wide
   shot with people further back). This is usually the biggest win.
2. **Lower `scale_factor`** (e.g. `1.1` → `1.05`) for finer-grained
   multi-scale steps — more precise boxes, at the cost of more compute
   per frame.
3. **Tune `min_neighbors`** — raise it if you're seeing false-positive
   boxes, lower it if a real face is being dropped entirely.
4. **Try `haarcascade_frontalface_default.xml`** instead of `_alt` — it's
   heavier but generally more robust with multiple/varied-scale faces.
   Both are already in `config.json` as `frontal_alt` and
   `frontal_default`. They're both enabled by default so you can compare
   detections side by side (alt's boxes are green, default's are
   orange), but running both doubles per-frame compute — once you've
   picked a winner for your footage, disable the other in `config.json`
   or run with `--cascades frontal_default` (or `frontal_alt`).
5. If Haar cascades remain unreliable for your footage even after
   tuning, that's an inherent ceiling of this detector family (no
   rotation/scale invariance, weak under occlusion) — set
   `"detector": "dnn"` (or run with `--detector dnn`) to use the DNN face
   detector, which handles these cases far better at higher compute cost.

## The `videos/` directory

Local video files go in `videos/`. The directory itself is tracked (via
`.gitkeep`) but its contents are git-ignored, so sample/test footage never
gets committed to the repository. Any `.mp4`/`.avi`/`.mov`/`.mkv`/`.wmv`
file placed there is a candidate; how one gets picked depends on whether
you leave `video.path` empty or set it.

### Managing multiple video files

With more than one file in `videos/`, you have two strategies:

1. **Auto-discovery (`video.path: ""`)** — `app.py` picks the first file
   alphabetically and logs every other file it found, e.g.:
   ```
   Multiple video files found in videos/: clip_a.mp4, clip_b.mp4. Set
   video.path in config.json (or --video) to pick one explicitly; using
   'clip_a.mp4' (first alphabetically).
   ```
   This is convenient for a "drop one file in, just run it" workflow, but
   with several files it's implicit and order-dependent — fine for quick
   local testing, not for a reproducible/deployed setup.

2. **Explicit selection (recommended once you have more than one file)**
   — set `video.path` in `config.json` to the filename you want, e.g.:
   ```json
   "video": { "path": "clip_b.mp4", "camera_index": 0 }
   ```
   A bare filename like `"clip_b.mp4"` is resolved inside `videos/`
   automatically — no need to write `"videos/clip_b.mp4"` (though a full
   relative or absolute path also works if the file lives elsewhere).
   This is deterministic and self-documenting: anyone reading
   `config.json` knows exactly which footage a given deployment runs
   against.

For one-off runs without editing `config.json`, override per-invocation
with `--video clip_b.mp4`. If a configured/`--video` filename doesn't
exist, the warning lists every file actually present in `videos/`, so a
typo or a forgotten copy step is easy to spot:
```
Video path not found: ...\videos\clip_c.mp4. Falling back to system
camera. Available in videos/: clip_a.mp4, clip_b.mp4.
```

For managing several fixed video/cascade combinations (e.g. one per
camera or one per deployment site), keep a separate config file per
combination (`config.cam1.json`, `config.cam2.json`, …) each with its own
`video.path`, and select between them with `--config` — see
[Configuration](#configuration-configjson) above.

## Docker / edge deployment

The Dockerfile builds a slim, headless image intended for edge devices
(tested base image supports both x86_64 and ARM, e.g. Raspberry Pi).

**Build:**

```powershell
docker build -t face-detection .
```

**Run against a mounted video file (using config.json's cascade/detection settings):**

```powershell
docker run --rm -v ${PWD}\videos:/app/videos face-detection --no-display
```

**Run against a camera, passing the device through (Linux edge device):**

```bash
docker run --rm --device=/dev/video0 face-detection --camera --camera-index 0 --no-display
```

`--camera` forces the system camera even if a `videos/` volume happens to
be mounted too — otherwise `video.source: "auto"` would prefer any video
file it finds there.

**Run with an edge-specific config, without rebuilding the image:**

```bash
docker run --rm -v ${PWD}/config.edge.json:/app/config.json:ro --device=/dev/video0 face-detection --no-display
```

> Camera passthrough (`--device`) requires a Linux host and a Linux
> container; it is not available when running Docker Desktop's Windows/WSL
> backend against a Windows-attached webcam. Test camera passthrough
> directly on the target edge device.

The container has no display, so `--no-display` is set as the image's
default `CMD`. That only takes effect when you run the container with no
extra arguments (`docker run --rm face-detection`) — any arguments given
after the image name (as in the examples above) **replace** the default
`CMD` rather than adding to it, so include `--no-display` explicitly
whenever you also pass `--video` or `--camera-index`.

### Why `requirements-docker.txt` is separate

- `requirements.txt` installs `opencv-python`, which bundles GUI libraries
  needed for the local preview window (`cv2.imshow`).
- `requirements-docker.txt` installs `opencv-python-headless`, the same
  OpenCV build without the GUI dependencies — smaller image, and avoids
  missing shared-library errors (e.g. `libGL.so.1`) on a minimal base image.

## Known issue: `opencv-python` 5.0.x

`opencv-python==5.0.0.93` (the latest release on PyPI at the time of
writing) ships without `cv2.CascadeClassifier` in its Python bindings —
importing it and calling `cv2.CascadeClassifier(...)` raises
`AttributeError`. Both `requirements.txt` and `requirements-docker.txt`
pin OpenCV to the stable `4.10.0.84` release, which does not have this
problem. Re-test before upgrading past this pin.

## Controls (when a display is available)

- `q` or `Esc` — quit
- `Ctrl+C` in the terminal — quit
