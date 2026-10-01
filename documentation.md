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
├── models/                         # DNN face detector models (see "Models and data")
│   ├── face_detection_yunet_2026may.onnx          # active model
│   └── face_detection_yunet_2023mar_int8bq.onnx   # older quantized YuNet, not referenced by config/code
├── evaluation/                     # Latency/accuracy evaluation (see below)
│   ├── run_evaluation.py
│   ├── label_helper.py
│   ├── ground_truth.json
│   └── labeling/decisions.json
├── reports/                        # Evaluation output, one timestamped folder per run (git-ignored)
├── requirements.txt                # Dependencies for local/dev use (with GUI display)
├── requirements-docker.txt         # Dependencies for the container (headless)
├── requirements-eval.txt           # Extra dependency (matplotlib) for evaluation/
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
    { "name": "frontal_alt", "path": "cascades/haarcascade_frontalface_alt.xml", "enabled": true,
      "min_neighbors": 3 },
    { "name": "frontal_default", "path": "cascades/haarcascade_frontalface_default.xml", "enabled": true,
      "min_neighbors": 5, "min_weight": 2 }
  ],
  "dnn": {
    "model_path": "models/face_detection_yunet_2026may.onnx",
    "score_threshold": 0.7,
    "nms_threshold": 0.3,
    "top_k": 5000,
    "input_width": 0,
    "weak_score_threshold": 0.5,
    "enhance_contrast": false
  },
  "eyes": {
    "enabled": true,
    "source": "auto",
    "cascade_path": "cascades/haarcascade_eye.xml",
    "scale_factor": 1.1,
    "min_neighbors": 2,
    "min_size": [10, 10],
    "verify_haar_faces": false,
    "min_eyes": 1
  },
  "detection": {
    "scale_factor": 1.05,
    "min_neighbors": 3,
    "min_size": [24, 24],
    "max_size": null,
    "max_faces": 0,
    "min_cascade_agreement": 2,
    "detect_every_n_frames": 1
  },
  "tracking": {
    "enabled": true,
    "smoothing": 0.4,
    "face_hold_frames": { "haar": 1, "dnn": 0 },
    "eye_hold_frames": 15
  },
  "display": {
    "enabled": true,
    "show_eyes": false
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
  - `weak_score_threshold` (default `0.5`, `0` = off) — a face scoring
    between this and `score_threshold` is kept only if it continues a face
    that is already tracked; a new face still needs `score_threshold`. This
    stops a real face from blinking out when its confidence dips for a frame,
    without letting doubtful new boxes appear. Going lower than 0.5 mostly
    adds backs of heads and background (checked by eye on clip3).
  - `input_width` (default `0` = native) — frames wider than this are
    downscaled before the network and the boxes mapped back. Faster, but small
    faces are lost (see the change log); only use it when faces are large.
  - `enhance_contrast` (default `false`) — CLAHE on the luminance channel
    before detection. About 2x slower and no gain on the sample clips.
- **`eyes`** — eyes are found for every face (drawn as red boxes only when
  `display.show_eyes` is `true`). Set
  `enabled: false` to turn them off. `source` picks where they come from:
  `"landmarks"` uses the eye positions the DNN detector reports for each
  face (reliable, DNN only), `"cascade"` runs `haarcascade_eye.xml` inside
  the upper part of the face (at most one eye per side), and `"auto"`
  (default) uses landmarks with the DNN detector and the cascade with Haar.
  The eye cascade runs on the upper face band after a 2x upscale and CLAHE
  contrast enhancement; without that it found an eye in only ~21% of faces
  on the sample video (hazy, low contrast) and with it about 86%, at
  roughly 13 ms per face. `verify_haar_faces: true` (off by default) additionally
  discards Haar face boxes with fewer than `min_eyes` eyes; because of the
  cascade's low hit rate this also discards most real faces, and it can't
  work on small faces at all.
- **`tracking`** — removes box/eye flicker. Detectors miss a face or an
  eye on some frames and jitter on others, so detections are matched to
  faces from earlier frames and smoothed. `smoothing` (0–1) is the weight
  of the newest detection (lower = smoother but laggier), `face_hold_frames`
  keeps a face on screen for that many detection passes after it's lost
  (a number, or `{"haar": n, "dnn": n}` per detector), and `eye_hold_frames`
  does the same for each eye. Holding a lost face hides flicker but leaves
  a ghost box when a person leaves or turns away. Measured on the sample
  video: for DNN, hold 0 gives 0 false positives and the least flicker (3
  gives 5 false positives and twice the flicker), so DNN defaults to 0; for
  the flakier Haar detector hold 1 removes the ghost boxes (precision 95% →
  100%, F1 0.932 → 0.953) at some cost in face-count flicker, so Haar
  defaults to 1. Eyes are stored
  relative to their face box, so they follow it as it moves or grows. Set
  `enabled: false` to draw raw per-frame detections. On the sample video
  tracking cut frames where the eye count flickered from 96–105 to about 22
  and eye-box jumps (90th percentile) from about 50 px to about 6 px.
  Eye candidates from the cascade must also sit where eyes anatomically are
  on the face (about 22–56% down, left or right of centre, 12–42% of the
  face width) and the two eyes must be at least 32% of the face width
  apart; this rejects eyebrows, glasses frames and duplicate boxes on one
  eye.
- **`detection.max_faces`** — keep only the N largest faces per frame
  (`1` = only the face closest to the camera). `0` (default) keeps every
  face. Override with `--max-faces`. This selects by size, not identity: it
  doesn't recognise *whose* face it is.
- **`detection.min_cascade_agreement`** — (Haar only) how many enabled
  cascades must find a face for it to be kept. The **first** cascade scans
  the whole frame and proposes faces; every other cascade only re-checks a
  small region around each proposal (cheap) and votes if it finds an
  overlapping box (IoU > 0.3). With the two default cascades, `2` means
  both must agree. This is the main defence against boxes on
  clothing/bodies: a single cascade fires on textured clothing, but
  different cascades rarely make the same mistake. `1` uses the first
  cascade alone.
- **`detection.detect_every_n_frames`** — run detection only on every Nth
  frame and redraw the previous boxes in between. `1` (default) detects
  every frame. Raise it to cut CPU proportionally (Haar on the sample video
  takes about 100 s at `1` and 36 s at `3`); boxes lag by up to N-1 frames.
  Between detection passes the tracker moves each box on by its measured
  velocity, so boxes keep moving instead of freezing. Measured on the DNN
  (face-reco-video): at `2` the cost per frame roughly halves, precision and
  recall are unchanged, but box jitter rises (0.037 -> 0.095 of a face width
  per frame); at `5` recall falls to 0.84 and 14 false positives appear, so
  keep it at 1 or 2. Haar speed is otherwise governed mainly by `scale_factor` (1.05 is about
  1.6x slower than 1.08 but finds more faces) and `min_size`.
- **`cascades`** — (Haar only) an ordered list of cascades. Each entry has a `name`
  (used in logs and for `--cascades` overrides), a `path`, and an
  `enabled` flag. Set `enabled: false` to skip a cascade without deleting
  it from the file, or add a new entry to run a custom cascade you drop
  into `cascades/` — no code changes required. Optional per-cascade
  overrides: `min_neighbors` (replaces `detection.min_neighbors` for this
  cascade) and `min_weight` (this cascade only votes for boxes whose
  detection confidence reaches it). Merged faces use the first cascade's
  box and color. At least one cascade must be enabled. In Haar mode, a box
  lying directly below another face box (a torso under a head) is also
  dropped.
- **`detection.scale_factor` / `detection.min_neighbors`** —
  `detectMultiScale` parameters, applied to every enabled cascade
  (Haar only).
- **`detection.min_size` / `detection.max_size`** — (Haar only) `[width, height]` in
  pixels, or `null` to leave unconstrained. Bounds the face sizes
  `detectMultiScale` will consider; narrowing this range to match the
  actual face sizes in your footage reduces both false positives and
  stray/misaligned boxes. See the troubleshooting note below.
- **`display.enabled`** — whether to open a preview window.
- **`display.show_eyes`** — draw the red eye boxes on the frame (and the
  saved video). Default `false`: only face boxes are drawn. Eyes are still
  detected and tracked either way, so this only changes the drawing.

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
4. For every frame: detects faces with the configured detector (Haar:
   every enabled cascade, merged so a face must be confirmed by
   `min_cascade_agreement` cascades, with torso boxes under a head
   dropped; or the DNN detector), sorts them largest first, finds the
   eyes in each face, keeps at most `max_faces` (all by default), and draws
   the face box (green) plus red eye boxes.
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
| `--save-video`      | off                   | Also write the annotated frames to a video file (e.g. `out.mp4`). |
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

## Evaluating detector performance (`evaluation/`)

`evaluation/run_evaluation.py` runs the real pipeline (`app.Pipeline`, the
same class `app.py` uses) over a video once per detector, with per-frame
timing, and scores it against hand-reviewed ground truth. It writes graphs,
raw data and a written summary to a new, timestamped `reports/<date>-<time>/`
folder — nothing is overwritten between runs.

```powershell
.venv\Scripts\pip install -r requirements-eval.txt   # matplotlib, one-time
.venv\Scripts\python evaluation\run_evaluation.py
```

Each report folder contains:

- **`summary.md`** — the write-up: setup, latency table, accuracy table,
  eyes/stability table, embedded graphs, and a few auto-computed takeaways.
  Open this first.
- **PNG graphs** — latency over time, latency distribution and percentiles,
  where the time goes (detection/eyes/tracking) and throughput, latency vs.
  how many faces are in frame, two confusion matrices (frame-level "is
  there a face" and face-count), precision/recall/F1, recall by face size,
  eye coverage and output flicker, and an error gallery (missed faces and
  false-positive boxes, rendered on the actual frames).
- **`metrics.json`** — every number above, unrounded, plus the exact
  settings each detector ran with.
- **`per_frame.csv`** — one row per (detector, frame): latency broken down
  by stage, face count, eye count. For custom analysis/plots.

Options: `--detectors haar` or `--detectors dnn` to run one; `--iou` to
change the match threshold (default 0.3); `--repeats` for how many passes
to average per detector (default 3 — a machine hiccup on one pass, like a
background scan, is far less likely to skew 3 medians than 1 raw run);
`--config` to evaluate a different config file; `--video` for a different
clip (needs its own ground truth — see below).

### Confusion matrix note

A confusion matrix needs a defined negative class, which individual face
boxes don't have (there's no fixed number of "not a face" boxes a frame
could produce). Two matrices are reported instead: **frame-level** (does
this frame contain a face, yes/no) and **face-count** (0 / 1 / 2 / 3+ faces
in the frame, actual vs. predicted). Box-level correctness is reported the
usual way for detection instead: true/false positives/negatives and
precision/recall/F1, at a configurable IoU threshold.

### Ground truth (`evaluation/ground_truth.json`)

There's no independent labelled dataset for the sample video, so ground
truth is built semi-automatically: `evaluation/label_helper.py candidates`
samples every Nth frame, proposes candidate boxes from **both** detectors
at deliberately loose thresholds (so a real face is very unlikely to be
missed), and renders them onto review sheets under
`evaluation/labeling/sheets/`. A person looks at the sheets once and records
in `evaluation/labeling/decisions.json` which candidates are false positives
(plus any face nobody found, and any region too small/occluded to label
reliably). `label_helper.py build` then writes `ground_truth.json` from
those decisions. The sample video's `ground_truth.json` and
`decisions.json` are committed, so `run_evaluation.py` works out of the box;
re-run both steps if you swap in a different video.

## Models and data

Keep this section and the change log below up to date whenever the code's
behaviour, a model, or the data changes.

### Models

| File | Type | Status |
|---|---|---|
| `cascades/haarcascade_frontalface_alt.xml` | Haar, frontal face | Active (first cascade; proposes faces) |
| `cascades/haarcascade_frontalface_default.xml` | Haar, frontal face | Active (confirms faces, `min_weight` 2) |
| `cascades/haarcascade_eye.xml` | Haar, eye | Active for Haar eye detection |
| `cascades/haarcascade_profileface.xml` | Haar, left profile | Present, not enabled in `config.json` |
| `models/face_detection_yunet_2026may.onnx` | YuNet DNN (OpenCV `FaceDetectorYN`) | Active DNN model, also supplies eye landmarks |
| `models/face_detection_yunet_2023mar_int8bq.onnx` | YuNet DNN, int8 quantized | Present, but does not load in OpenCV 4.10 (`DequantizeLinear` error), so it cannot be used here |

### Video data (`videos/`, git-ignored, so not in the repository)

| Clip | Resolution / fps / frames | Ground truth | Notes |
|---|---|---|---|
| `face-reco-video.mp4` | 768x432, 12 fps, 1091 frames | Yes (`evaluation/ground_truth.json`, 91 frames, 90 faces) | Used for all numbers in the existing report. |
| `clip2-detection.mp4` | not profiled | No | |
| `clip3-detection.mp4` | 848x478, 30 fps, 530 frames | No | Crowd scene with small and turned faces; see change log. |

## Change log and findings

### 2026-10-01 - DNN speed and stability work
- Code (`app.py`): new `DnnFaceDetector` wrapper (optional downscale,
  optional CLAHE, only calls `setInputSize` when the size changes);
  `dnn.weak_score_threshold` tier applied in `Pipeline._confident` using
  `FaceTracker.has_track_near`; boxes coast on measured velocity between
  detection passes (`FaceTrack.velocity`, `snapshot(coast)`); the grayscale
  conversion is skipped when neither Haar nor the eye cascade needs it.
  `detect_faces_dnn` now returns a 7th item, the score.
- Config (`config.json`, defaults in `app.py`): added `dnn.input_width` (0),
  `dnn.weak_score_threshold` (0.5), `dnn.enhance_contrast` (false).
- Results (DNN; face-reco-video with ground truth, and clip3 without):
  - Weak tier 0.5: no change on the labelled clip (precision 1.000, recall
    0.989, 0 false positives); on clip3 face-count changes fell 295 -> 225
    and detections rose 1724 -> 2201. By eye, the added boxes are mostly real
    faces of people turned away or small; below 0.5 they are mostly junk.
  - Face hold: hold 1 gave 3 false positives and flicker 16 -> 36, hold 2 gave
    4 and 28. This confirms the existing note, so the DNN hold stays 0.
  - Downscale: 640 px was about 20% faster but found 29% fewer faces on clip3
    (1724 -> 1222); 512 px about 2x faster, 2 of 90 labelled faces lost and
    52% fewer on clip3. Left off by default.
  - CLAHE: 65 ms against 36 ms, no accuracy gain. Left off.
  - Adaptive smoothing (heavier smoothing on still faces): jitter only
    0.037 -> 0.035 and box overlap fell slightly. Tried and removed.
  - Same-session A/B of the old and new code: no slowdown (new 38.4-44.6 ms
    against old 38.9-48.4 ms). Absolute timings on this machine drift by
    +-30% between runs, so compare only interleaved runs.
- Display: added `display.show_eyes` (default `false`) so only face boxes are
  drawn; set it to `true` to bring the red eye boxes back. Detection,
  tracking and the evaluation's eye statistics are unaffected.
- Not yet done: ground truth for clip3, so its numbers above are counts and
  flicker only, not precision/recall.


### 2026-10-01 - clip3: Haar is not usable on crowd footage
- Running `detector: haar` on `clip3-detection.mp4` (every 10th frame, 53
  frames): Haar found 1 face in total, against 180 for the DNN, at about
  255 ms per frame against 58 ms for the DNN (same machine).
- Looser Haar settings (one cascade only, `min_neighbors` 2, `scale_factor`
  1.1, `min_size` 20) took 120-230 ms per frame and still matched at most 4
  of the 180 DNN faces. Haar cannot be tuned to cope with this clip; it
  fails on small, turned and partly hidden faces and slows down with every
  scale it scans.
- Recommendation: use `"detector": "dnn"` for crowd footage. Haar remains
  available for comparison.
- Config note: the working-tree `config.json` was switched to `detector:
  "haar"` and `video.path: "videos/clip3-detection.mp4"` for this test; the
  committed default is `dnn` with an empty video path.
- No code was changed for this finding. To score clip3 properly, ground
  truth must first be built with `evaluation/label_helper.py`.

### 2026-09-28 - Initial DNN + evaluation work (commit 8889bdf)
- Added the YuNet DNN detector and the `detector` switch, DNN eye landmarks,
  multi-cascade agreement voting, torso-box suppression, face/eye tracking
  and smoothing, and `detect_every_n_frames`.
- Added `evaluation/` (latency and accuracy report, labelling helper,
  ground truth for `face-reco-video.mp4`).
- Results on `face-reco-video.mp4`: Haar recall 0.911 and 78 ms per frame;
  DNN recall 0.989 and 22 ms per frame (report `reports/20260929-095747`).

## Controls (when a display is available)

- `q` or `Esc` — quit
- `Ctrl+C` in the terminal — quit
