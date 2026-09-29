"""Latency + accuracy evaluation of the Haar and DNN detectors.

Runs the real app.Pipeline (face detection, eyes, tracking) over the whole
video once per detector, timing every stage on every frame, and scores the
frames that have ground truth (evaluation/ground_truth.json). Everything is
written to reports/<timestamp>/: graphs (PNG), metrics.json, per_frame.csv
and summary.md. Re-running never overwrites an earlier report.

    python evaluation/run_evaluation.py
    python evaluation/run_evaluation.py --detectors dnn --iou 0.5
    python evaluation/run_evaluation.py --config config.edge.json

Needs matplotlib: pip install -r requirements-eval.txt
"""

import argparse
import csv
import datetime
import json
import logging
import os
import platform
import sys
from collections import defaultdict

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EVAL_DIR)
sys.path.insert(0, ROOT)

import app  # noqa: E402  (needs ROOT on sys.path)

GROUND_TRUTH_PATH = os.path.join(EVAL_DIR, "ground_truth.json")
REPORTS_DIR = os.path.join(ROOT, "reports")

COLORS = {"haar": "#d97706", "dnn": "#2563eb"}
LABELS = {"haar": "Haar cascades", "dnn": "DNN (YuNet)"}
STAGES = [("detect_ms", "Face detection"), ("eyes_ms", "Eye detection"), ("track_ms", "Tracking")]
STAGE_COLORS = ["#4b5563", "#059669", "#a855f7"]
SIZE_BUCKETS = [(0, 40, "< 40 px"), (40, 80, "40-80 px"), (80, 10_000, ">= 80 px")]


# --------------------------------------------------------------------------- running


def build_pipeline(detector: str, config: str | None) -> "app.Pipeline":
    """Create the app's Pipeline for one detector using the normal config resolution."""
    argv = ["app.py", "--detector", detector, "--no-display"]
    if config:
        argv += ["--config", config]
    old_argv = sys.argv
    sys.argv = argv
    try:
        settings = app.resolve_settings(app.parse_args())
    finally:
        sys.argv = old_argv
    return app.Pipeline(settings), settings


def run_detector(detector: str, video: str, config: str | None, warmup: int) -> dict:
    pipeline, settings = build_pipeline(detector, config)
    cap = cv2.VideoCapture(video)
    frames = []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        start = app.time.perf_counter()
        output = pipeline.process(frame)
        total_ms = (app.time.perf_counter() - start) * 1000
        eyes_found = sum(e is not None for _, eyes in output for e in eyes)
        frames.append({
            "frame": index,
            "total_ms": total_ms,
            **pipeline.timings,
            "faces": [list(face[:4]) for face, _ in output],
            "n_faces": len(output),
            "n_eyes": eyes_found,
            "eyes_per_face": [sum(e is not None for e in eyes) for _, eyes in output],
        })
        index += 1
    cap.release()
    print("  %-4s processed %d frames, mean %.1f ms/frame" % (
        detector, len(frames), np.mean([f["total_ms"] for f in frames[warmup:]])))
    return {"frames": frames, "settings": settings, "warmup": warmup}


# --------------------------------------------------------------------------- scoring


def in_regions(box, regions) -> bool:
    cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
    return any(r[0] <= cx <= r[0] + r[2] and r[1] <= cy <= r[1] + r[3] for r in regions)


def match(predictions: list, truths: list, iou_threshold: float):
    """Greedy one-to-one matching by IoU. Returns (matches, unmatched preds, unmatched truths)."""
    pairs = sorted(
        ((app.box_iou(p, t), pi, ti) for pi, p in enumerate(predictions) for ti, t in enumerate(truths)),
        reverse=True,
    )
    used_p, used_t, matches = set(), set(), []
    for overlap, pi, ti in pairs:
        if overlap < iou_threshold:
            break
        if pi in used_p or ti in used_t:
            continue
        used_p.add(pi)
        used_t.add(ti)
        matches.append((pi, ti, overlap))
    return (
        matches,
        [pi for pi in range(len(predictions)) if pi not in used_p],
        [ti for ti in range(len(truths)) if ti not in used_t],
    )


def score(run: dict, truth: dict, iou_threshold: float) -> dict:
    frames = run["frames"]
    tp = fp = fn = 0
    ious = []
    frame_cm = np.zeros((2, 2), dtype=int)  # rows: actual [no face, face]; cols: predicted
    count_cm = np.zeros((4, 4), dtype=int)  # rows: actual count 0..3+; cols: predicted count 0..3+
    size_hits = {label: [0, 0] for _, _, label in SIZE_BUCKETS}  # [found, total]
    errors = []  # every false positive / false negative, for the error gallery

    for key, truths in truth["faces_per_frame"].items():
        index = int(key)
        if index >= len(frames):
            continue
        ignore = truth.get("ignore_regions", {}).get(key, [])
        predictions = frames[index]["faces"]
        matches, extra_preds, missed = match(predictions, truths, iou_threshold)
        extra_preds = [pi for pi in extra_preds if not in_regions(predictions[pi], ignore)]

        tp += len(matches)
        fp += len(extra_preds)
        fn += len(missed)
        ious += [m[2] for m in matches]
        errors += [{"frame": index, "kind": "FP", "box": predictions[pi]} for pi in extra_preds]
        errors += [{"frame": index, "kind": "FN", "box": truths[ti]} for ti in missed]

        kept_predictions = len(matches) + len(extra_preds)
        frame_cm[int(len(truths) > 0), int(kept_predictions > 0)] += 1
        count_cm[min(len(truths), 3), min(kept_predictions, 3)] += 1

        found_truths = {ti for _, ti, _ in matches}
        for ti, box in enumerate(truths):
            for low, high, label in SIZE_BUCKETS:
                if low <= box[2] < high:
                    size_hits[label][1] += 1
                    size_hits[label][0] += ti in found_truths

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "frame_confusion": frame_cm.tolist(),
        "count_confusion": count_cm.tolist(),
        "recall_by_size": {k: {"found": v[0], "total": v[1]} for k, v in size_hits.items()},
        "errors": errors,
    }


def merge_repeats(passes: list[dict]) -> dict:
    """Combine repeated passes over the video into one run.

    Detections are deterministic, so the first pass supplies the faces and
    counts; each frame's timings become the median across passes, which
    removes one-off slowdowns from other processes on the machine.
    """
    merged = passes[0]
    if len(passes) > 1:
        for i, frame in enumerate(merged["frames"]):
            for key in ("total_ms", "detect_ms", "eyes_ms", "track_ms"):
                frame[key] = float(np.median([p["frames"][i][key] for p in passes]))
    means = [np.mean([f["total_ms"] for f in p["frames"][p["warmup"]:]]) for p in passes]
    merged["pass_means_ms"] = [float(m) for m in means]
    return merged


def latency_stats(run: dict) -> dict:
    frames = run["frames"][run["warmup"]:]
    total = np.array([f["total_ms"] for f in frames])
    stats = {
        "pass_means_ms": run.get("pass_means_ms", []),
        "frames": len(total),
        "mean_ms": float(total.mean()), "median_ms": float(np.median(total)),
        "std_ms": float(total.std()), "min_ms": float(total.min()),
        "p95_ms": float(np.percentile(total, 95)), "p99_ms": float(np.percentile(total, 99)),
        "max_ms": float(total.max()),
        "fps_mean": float(1000 / total.mean()),
        "stage_mean_ms": {k: float(np.mean([f[k] for f in frames])) for k, _ in STAGES},
    }
    by_faces = defaultdict(list)
    for f in frames:
        by_faces[min(f["n_faces"], 3)].append(f["total_ms"])
    stats["by_predicted_faces"] = {
        str(k): {"frames": len(v), "mean_ms": float(np.mean(v)), "p95_ms": float(np.percentile(v, 95))}
        for k, v in sorted(by_faces.items())
    }
    return stats


def stability_stats(run: dict) -> dict:
    """Ground-truth-free behaviour: eye coverage and how often the output flickers."""
    frames = run["frames"]
    face_frames = [f for f in frames if f["n_faces"]]
    n_faces_total = sum(f["n_faces"] for f in face_frames)
    both = sum(sum(1 for e in f["eyes_per_face"] if e == 2) for f in face_frames)
    at_least_one = sum(sum(1 for e in f["eyes_per_face"] if e >= 1) for f in face_frames)
    count_changes = sum(1 for a, b in zip(frames, frames[1:]) if a["n_faces"] != b["n_faces"])
    eye_flips = sum(
        1 for a, b in zip(frames, frames[1:]) if a["n_faces"] == b["n_faces"] and a["n_eyes"] != b["n_eyes"]
    )
    return {
        "face_observations": n_faces_total,
        "faces_with_both_eyes_pct": 100 * both / n_faces_total if n_faces_total else 0.0,
        "faces_with_any_eye_pct": 100 * at_least_one / n_faces_total if n_faces_total else 0.0,
        "face_count_changes": count_changes,
        "eye_count_flips": eye_flips,
        "frames": len(frames),
    }


# --------------------------------------------------------------------------- plots


def style_axes(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def legend_top(ax, loc="upper right", **kwargs):
    """Legend in a strip of headroom made above the data, so it never covers a bar."""
    low, high = ax.get_ylim()
    ax.set_ylim(low, high * 1.2)
    ax.legend(frameon=False, loc=loc, ncol=4, **kwargs)


def save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return name


def plot_latency_timeline(runs, out_dir, budget_ms):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    window = 15
    for name, run in runs.items():
        y = np.array([f["total_ms"] for f in run["frames"]])
        ax.plot(y, color=COLORS[name], alpha=0.22, linewidth=0.8)
        smooth = np.convolve(y, np.ones(window) / window, mode="same")
        ax.plot(smooth, color=COLORS[name], linewidth=2, label="%s (rolling mean, %d frames)" % (LABELS[name], window))
    ax.axhline(budget_ms, color="#dc2626", linestyle="--", linewidth=1,
               label="Real-time budget at the video's fps (%.0f ms)" % budget_ms)
    style_axes(ax, "Per-frame latency over the video", "Frame", "Latency (ms)")
    legend_top(ax, loc="upper left", fontsize=9)
    return save(fig, out_dir, "latency_timeline.png")


def plot_latency_distribution(runs, stats, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    upper = max(stats[n]["p99_ms"] for n in runs) * 1.1
    bins = np.linspace(0, upper, 50)
    for name, run in runs.items():
        y = [f["total_ms"] for f in run["frames"][run["warmup"]:]]
        ax.hist(y, bins=bins, color=COLORS[name], alpha=0.6, label=LABELS[name])
        ax.axvline(stats[name]["p95_ms"], color=COLORS[name], linestyle=":", linewidth=1.5)
    style_axes(ax, "Latency distribution (dotted line = p95)", "Latency per frame (ms)", "Frames")
    legend_top(ax)

    ax = axes[1]
    names = list(runs)
    metrics = [("median_ms", "median"), ("p95_ms", "p95"), ("p99_ms", "p99")]
    x = np.arange(len(metrics))
    width = 0.8 / len(names)
    for i, name in enumerate(names):
        vals = [stats[name][k] for k, _ in metrics]
        bars = ax.bar(x + i * width - 0.4 + width / 2, vals, width, color=COLORS[name], label=LABELS[name])
        ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=2)
    ax.set_xticks(x, [label for _, label in metrics])
    style_axes(ax, "Latency percentiles", None, "ms")
    return save(fig, out_dir, "latency_distribution.png")


def plot_latency_breakdown(runs, stats, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    names = list(runs)
    bottom = np.zeros(len(names))
    for (key, label), color in zip(STAGES, STAGE_COLORS):
        vals = np.array([stats[n]["stage_mean_ms"][key] for n in names])
        bars = ax.bar([LABELS[n] for n in names], vals, bottom=bottom, color=color, label=label, width=0.55)
        for b, v, base in zip(bars, vals, bottom):
            if v > 1.5:
                ax.text(b.get_x() + b.get_width() / 2, base + v / 2, "%.1f" % v, ha="center", va="center",
                        color="white", fontsize=9)
        bottom += vals
    style_axes(ax, "Where the time goes (mean ms per frame)", None, "ms per frame")
    legend_top(ax)

    ax = axes[1]
    fps = [stats[n]["fps_mean"] for n in names]
    bars = ax.bar([LABELS[n] for n in names], fps, color=[COLORS[n] for n in names], width=0.55)
    ax.bar_label(bars, fmt="%.1f fps", padding=3)
    style_axes(ax, "Throughput (mean frames per second)", None, "fps")
    return save(fig, out_dir, "latency_breakdown_throughput.png")


def plot_latency_by_occupancy(runs, stats, out_dir):
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    keys = ["0", "1", "2", "3"]
    x = np.arange(len(keys))
    width = 0.38
    for i, name in enumerate(runs):
        data = stats[name]["by_predicted_faces"]
        means = [data.get(k, {}).get("mean_ms", 0) for k in keys]
        p95 = [data.get(k, {}).get("p95_ms", 0) for k in keys]
        bars = ax.bar(x + (i - 0.5) * width, means, width, color=COLORS[name], label=LABELS[name] + " (mean)")
        ax.scatter(x + (i - 0.5) * width, p95, color="black", s=14, zorder=3,
                   label="p95" if i == 0 else None)
        ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=2)
    labels = []
    for k in keys:
        frames_each = ", ".join("%s %d" % (n.upper() if n == "dnn" else n.capitalize(),
                                           stats[n]["by_predicted_faces"].get(k, {}).get("frames", 0))
                                for n in runs)
        labels.append("%s\nframes: %s" % (("3+" if k == "3" else k) + " face" + ("" if k == "1" else "s"), frames_each))
    ax.set_xticks(x, labels, fontsize=9)
    style_axes(ax, "Latency vs. how many faces are in the frame", "Faces each detector found in the frame", "ms per frame")
    legend_top(ax)
    return save(fig, out_dir, "latency_by_face_count.png")


def draw_matrix(ax, matrix, xlabels, ylabels, title):
    matrix = np.array(matrix)
    row_sums = matrix.sum(axis=1, keepdims=True)
    shares = np.divide(matrix, row_sums, out=np.zeros(matrix.shape, dtype=float), where=row_sums > 0)
    ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if shares[i, j] > 0.55 else "#111827"
            ax.text(j, i, "%d\n(%.0f%%)" % (matrix[i, j], 100 * shares[i, j]), ha="center", va="center",
                    color=color, fontsize=11, fontweight="bold")
    ax.set_xticks(range(matrix.shape[1]), xlabels)
    ax.set_yticks(range(matrix.shape[0]), ylabels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual (ground truth)")
    ax.set_title(title, fontsize=12, fontweight="bold")


def plot_confusion(runs, scores, out_dir):
    names = list(runs)
    fig, axes = plt.subplots(1, len(names), figsize=(5.6 * len(names), 4.8))
    for ax, name in zip(np.atleast_1d(axes), names):
        draw_matrix(ax, scores[name]["frame_confusion"], ["No face", "Face"], ["No face", "Face"],
                    LABELS[name])
    fig.suptitle("Frame-level confusion matrix (does the frame contain a face?)", fontsize=13,
                 fontweight="bold", y=1.02)
    fig.text(0.5, -0.03, "Cells show frame counts and the share of each actual class (row).",
             ha="center", fontsize=9, color="#4b5563")
    a = save(fig, out_dir, "confusion_matrix_frame.png")

    fig, axes = plt.subplots(1, len(names), figsize=(5.8 * len(names), 5))
    ticks = ["0", "1", "2", "3+"]
    for ax, name in zip(np.atleast_1d(axes), names):
        draw_matrix(ax, scores[name]["count_confusion"], ticks, ticks, LABELS[name])
    fig.suptitle("Face-count confusion matrix (number of faces in the frame)", fontsize=13,
                 fontweight="bold", y=1.02)
    b = save(fig, out_dir, "confusion_matrix_face_count.png")
    return a, b


def plot_detection_quality(runs, scores, iou, out_dir):
    names = list(runs)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    metrics = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]
    x = np.arange(len(metrics))
    width = 0.38
    for i, name in enumerate(names):
        vals = [scores[name][k] * 100 for k, _ in metrics]
        bars = ax.bar(x + (i - 0.5) * width, vals, width, color=COLORS[name], label=LABELS[name])
        ax.bar_label(bars, fmt="%.1f%%", fontsize=9, padding=2)
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylim(0, 112)
    style_axes(ax, "Face detection quality (IoU >= %.2f)" % iou, None, "%")
    legend_top(ax)

    ax = axes[1]
    counts = [("tp", "True positives\n(correct faces)", "#059669"), ("fp", "False positives\n(boxes that aren't faces)", "#dc2626"),
              ("fn", "False negatives\n(missed faces)", "#6b7280")]
    x = np.arange(len(counts))
    for i, name in enumerate(names):
        vals = [scores[name][k] for k, _, _ in counts]
        bars = ax.bar(x + (i - 0.5) * width, vals, width, color=COLORS[name], label=LABELS[name])
        ax.bar_label(bars, fmt="%d", fontsize=9, padding=2)
    ax.set_xticks(x, [label for _, label, _ in counts])
    style_axes(ax, "Face box counts (labelled frames)", None, "Boxes")
    return save(fig, out_dir, "detection_quality.png")


def plot_recall_by_size(runs, scores, out_dir):
    names = list(runs)
    labels = [label for _, _, label in SIZE_BUCKETS]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(labels))
    width = 0.38
    for i, name in enumerate(names):
        vals = []
        for label in labels:
            d = scores[name]["recall_by_size"][label]
            vals.append(100 * d["found"] / d["total"] if d["total"] else 0)
        bars = ax.bar(x + (i - 0.5) * width, vals, width, color=COLORS[name], label=LABELS[name])
        ax.bar_label(bars, fmt="%.0f%%", fontsize=9, padding=2)
    totals = [scores[names[0]]["recall_by_size"][label]["total"] for label in labels]
    ax.set_xticks(x, ["%s\n(%d faces)" % (label, t) for label, t in zip(labels, totals)])
    ax.set_ylim(0, 112)
    style_axes(ax, "Recall by face width (small faces are harder)", "Ground-truth face width", "% of faces found")
    legend_top(ax)
    return save(fig, out_dir, "recall_by_face_size.png")


def plot_eyes(runs, stability, out_dir):
    names = list(runs)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    ax = axes[0]
    metrics = [("faces_with_both_eyes_pct", "Both eyes"), ("faces_with_any_eye_pct", "At least one eye")]
    x = np.arange(len(metrics))
    width = 0.38
    for i, name in enumerate(names):
        vals = [stability[name][k] for k, _ in metrics]
        bars = ax.bar(x + (i - 0.5) * width, vals, width, color=COLORS[name], label=LABELS[name])
        ax.bar_label(bars, fmt="%.0f%%", fontsize=9, padding=2)
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylim(0, 112)
    style_axes(ax, "Eye coverage on detected faces", None, "% of detected faces")
    legend_top(ax)

    ax = axes[1]
    metrics = [("face_count_changes", "Face count changed\nbetween frames"), ("eye_count_flips", "Eye count changed\n(same faces)")]
    x = np.arange(len(metrics))
    for i, name in enumerate(names):
        vals = [stability[name][k] for k, _ in metrics]
        bars = ax.bar(x + (i - 0.5) * width, vals, width, color=COLORS[name], label=LABELS[name])
        ax.bar_label(bars, fmt="%d", fontsize=9, padding=2)
    ax.set_xticks(x, [label for _, label in metrics])
    style_axes(ax, "Output flicker (lower is steadier)", None, "Frames (of %d)" % stability[names[0]]["frames"])
    return save(fig, out_dir, "eye_and_stability.png")


def plot_error_examples(runs, scores, truth, video, out_dir, per_detector=8):
    """Gallery of each detector's mistakes: green = ground truth, red = the error itself."""
    cap = cv2.VideoCapture(video)
    names = list(runs)
    cols = 4
    fig, axes = plt.subplots(len(names), cols, figsize=(4.4 * cols, 2.7 * len(names) + 0.9), squeeze=False)
    for row, name in enumerate(names):
        by_frame = defaultdict(list)
        for e in scores[name]["errors"]:
            by_frame[e["frame"]].append(e)
        shown = sorted(by_frame)[:per_detector]
        for col in range(cols):
            ax = axes[row][col]
            ax.axis("off")
            if col >= len(shown):
                continue
            frame_index = shown[col]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, image = cap.read()
            if not ok:
                continue
            for box in truth["faces_per_frame"].get(str(frame_index), []):
                cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 200, 0), 2)
            for e in by_frame[frame_index]:
                x, y, w, h = e["box"]
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 230), 3)
                cv2.putText(image, e["kind"], (x, max(y - 6, 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 230), 2)
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            kinds = "/".join(sorted({e["kind"] for e in by_frame[frame_index]}))
            ax.set_title("frame %d (%s)" % (frame_index, kinds), fontsize=10)
        axes[row][0].text(-0.02, 0.5, LABELS[name], transform=axes[row][0].transAxes, rotation=90,
                          va="center", ha="right", fontsize=12, fontweight="bold", color=COLORS[name])
        total = len(by_frame)
        if total > per_detector:
            axes[row][cols - 1].text(1.0, -0.05, "+%d more error frames (see metrics.json)" % (total - per_detector),
                                     transform=axes[row][cols - 1].transAxes, ha="right", fontsize=8)
    cap.release()
    fig.suptitle("Error examples: FP = box that isn't a face, FN = missed face (green = ground truth)",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout()
    return save(fig, out_dir, "error_examples.png")


def key_findings(names, stats, scores, stability) -> list[str]:
    """A few plain-language takeaways computed from the numbers."""
    out = []
    if "haar" in names and "dnn" in names:
        h, d = stats["haar"], stats["dnn"]
        out.append("DNN is %.1fx faster than Haar on average (%.1f vs %.1f ms/frame; p95 %.0f vs %.0f ms)." % (
            h["mean_ms"] / d["mean_ms"], d["mean_ms"], h["mean_ms"], d["p95_ms"], h["p95_ms"]))
        hq, dq = scores["haar"], scores["dnn"]
        out.append("Recall: DNN %.1f%% vs Haar %.1f%%; precision: DNN %.1f%% vs Haar %.1f%%; F1: %.3f vs %.3f." % (
            100 * dq["recall"], 100 * hq["recall"], 100 * dq["precision"], 100 * hq["precision"], dq["f1"], hq["f1"]))
        out.append("Haar latency grows with the number of faces in view (%.0f ms with none, %.0f ms with two); "
                   "DNN stays nearly flat (%.0f vs %.0f ms)." % (
                       h["by_predicted_faces"].get("0", {}).get("mean_ms", 0),
                       h["by_predicted_faces"].get("2", {}).get("mean_ms", 0),
                       d["by_predicted_faces"].get("0", {}).get("mean_ms", 0),
                       d["by_predicted_faces"].get("2", {}).get("mean_ms", 0)))
        out.append("Eyes on detected faces: both eyes %.0f%% (DNN landmarks) vs %.0f%% (Haar eye cascade)." % (
            stability["dnn"]["faces_with_both_eyes_pct"], stability["haar"]["faces_with_both_eyes_pct"]))
    return out


# --------------------------------------------------------------------------- reporting


def write_csv(runs, out_dir):
    path = os.path.join(out_dir, "per_frame.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["detector", "frame", "total_ms", "detect_ms", "eyes_ms", "track_ms", "n_faces", "n_eyes"])
        for name, run in runs.items():
            for fr in run["frames"]:
                writer.writerow([name, fr["frame"], "%.3f" % fr["total_ms"], "%.3f" % fr["detect_ms"],
                                 "%.3f" % fr["eyes_ms"], "%.3f" % fr["track_ms"], fr["n_faces"], fr["n_eyes"]])


def write_summary(out_dir, meta, runs, stats, scores, stability, images, iou):
    names = list(runs)
    L = lambda n: LABELS[n]  # noqa: E731
    lines = [
        "# Face detection evaluation report", "",
        "Generated %s" % meta["timestamp"], "",
        "## Setup", "",
        "- Video: `%s` - %d frames, %s, %.1f fps" % (meta["video"], meta["video_frames"], meta["resolution"], meta["video_fps"]),
        "- Machine: %s, %d logical CPUs, Python %s, OpenCV %s" % (meta["cpu"], meta["cpu_count"], meta["python"], meta["opencv"]),
        "- Ground truth: %d frames labelled (every %d-th frame), %d faces; matching IoU >= %.2f. "
        "Labels were made by reviewing candidate boxes by eye (see `evaluation/label_helper.py`)." % (
            meta["gt_frames"], meta["gt_step"], meta["gt_faces"], iou),
        "- Latency = full pipeline per frame (face detection + eyes + tracking), video decoding excluded, "
        "first %d frames dropped as warm-up. Each detector ran %d pass(es) over the video, one after the other; "
        "each frame's latency is the median across passes." % (meta["warmup"], meta["repeats"]),
        "", "## Latency", "",
        "| | " + " | ".join(L(n) for n in names) + " |", "|---|" + "---|" * len(names),
    ]
    rows = [("Mean (ms)", "mean_ms", "%.1f"), ("Median (ms)", "median_ms", "%.1f"), ("p95 (ms)", "p95_ms", "%.1f"),
            ("p99 (ms)", "p99_ms", "%.1f"), ("Max (ms)", "max_ms", "%.1f"), ("Throughput (fps)", "fps_mean", "%.1f")]
    for label, key, fmt in rows:
        lines.append("| %s | %s |" % (label, " | ".join(fmt % stats[n][key] for n in names)))
    for key, label in STAGES:
        lines.append("| Mean %s (ms) | %s |" % (label.lower(), " | ".join("%.1f" % stats[n]["stage_mean_ms"][key] for n in names)))
    lines += ["", "![timeline](%s)" % images["timeline"], "", "![distribution](%s)" % images["distribution"], "",
              "![breakdown](%s)" % images["breakdown"], "", "![by face count](%s)" % images["occupancy"], "",
              "## Accuracy (labelled frames)", "",
              "| | " + " | ".join(L(n) for n in names) + " |", "|---|" + "---|" * len(names)]
    for label, key, fmt in [("True positives", "tp", "%d"), ("False positives", "fp", "%d"), ("False negatives", "fn", "%d"),
                            ("Precision", "precision", "%.3f"), ("Recall", "recall", "%.3f"), ("F1", "f1", "%.3f"),
                            ("Mean IoU of correct boxes", "mean_iou", "%.3f")]:
        lines.append("| %s | %s |" % (label, " | ".join(fmt % scores[n][key] for n in names)))
    lines += ["", "![quality](%s)" % images["quality"], "", "![frame confusion](%s)" % images["cm_frame"], "",
              "![count confusion](%s)" % images["cm_count"], "", "![recall by size](%s)" % images["size"], "",
              "## Eyes and stability (no ground truth needed)", "",
              "| | " + " | ".join(L(n) for n in names) + " |", "|---|" + "---|" * len(names)]
    for label, key, fmt in [("Faces with both eyes", "faces_with_both_eyes_pct", "%.1f%%"),
                            ("Faces with at least one eye", "faces_with_any_eye_pct", "%.1f%%"),
                            ("Frames where face count changed", "face_count_changes", "%d"),
                            ("Frames where eye count flipped", "eye_count_flips", "%d")]:
        lines.append("| %s | %s |" % (label, " | ".join(fmt % stability[n][key] for n in names)))
    lines += ["", "![eyes](%s)" % images["eyes"], "",
              "## Mistakes", "", "![error examples](%s)" % images["errors"], "",
              "## Reading these results", "",]
    lines += ["- " + t for t in key_findings(names, stats, scores, stability)]
    lines += [
              "- A confusion matrix needs a definition of a negative. Here it is computed two ways: per frame (face present or not) "
              "and per face count. Individual boxes have no true negatives, so box-level results are TP/FP/FN with precision/recall.",
              "- The ground truth is small (%d faces) and hand-reviewed, so treat differences of a few percent as noise." % meta["gt_faces"],
              "- Latency depends on the machine and on how many faces are in the frame; compare runs on the same machine.", ""]
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detectors", default="haar,dnn", help="Comma-separated: haar, dnn (default both).")
    parser.add_argument("--config", default=None, help="Config file to evaluate (default: config.json).")
    parser.add_argument("--video", default=None, help="Video to run (default: first file in videos/).")
    parser.add_argument("--iou", type=float, default=0.3, help="IoU needed to count a box as matching (default 0.3).")
    parser.add_argument("--warmup", type=int, default=5, help="Frames excluded from latency stats (default 5).")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Passes over the video per detector; latency is the per-frame median (default 3).")
    parser.add_argument("--output", default=REPORTS_DIR, help="Directory reports are written under.")
    args = parser.parse_args()

    logging.getLogger("face_detection").setLevel(logging.WARNING)
    names = [n.strip() for n in args.detectors.split(",") if n.strip()]
    if not all(n in app.VALID_DETECTORS for n in names):
        sys.exit("--detectors must be from %s" % (app.VALID_DETECTORS,))

    video = args.video or app.discover_video(app.VIDEOS_DIR)
    if not video or not os.path.isfile(video):
        sys.exit("No video found (put one in videos/ or pass --video).")
    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        truth = json.load(f)

    cap = cv2.VideoCapture(video)
    meta = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "video": os.path.basename(video),
        "video_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "video_fps": cap.get(cv2.CAP_PROP_FPS) or 0.0,
        "resolution": "%dx%d" % (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "gt_frames": len(truth["faces_per_frame"]),
        "gt_step": truth["frame_step"],
        "gt_faces": sum(len(v) for v in truth["faces_per_frame"].values()),
        "warmup": args.warmup,
        "iou": args.iou,
        "detectors": names,
    }
    cap.release()
    if os.path.basename(video) != truth["video"]:
        print("Warning: ground truth was labelled on %s but evaluating %s." % (truth["video"], meta["video"]))

    meta["repeats"] = args.repeats
    print("Running %s over %s (%d pass%s each) ..." % (
        ", ".join(names), meta["video"], args.repeats, "" if args.repeats == 1 else "es"))
    runs = {}
    for name in names:
        passes = [run_detector(name, video, args.config, args.warmup) for _ in range(args.repeats)]
        runs[name] = merge_repeats(passes)

    stats = {n: latency_stats(r) for n, r in runs.items()}
    scores = {n: score(r, truth, args.iou) for n, r in runs.items()}
    stability = {n: stability_stats(r) for n, r in runs.items()}

    out_dir = os.path.join(args.output, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir)
    budget = 1000 / meta["video_fps"] if meta["video_fps"] else 40.0
    cm_frame, cm_count = plot_confusion(runs, scores, out_dir)
    images = {
        "timeline": plot_latency_timeline(runs, out_dir, budget),
        "distribution": plot_latency_distribution(runs, stats, out_dir),
        "breakdown": plot_latency_breakdown(runs, stats, out_dir),
        "occupancy": plot_latency_by_occupancy(runs, stats, out_dir),
        "quality": plot_detection_quality(runs, scores, args.iou, out_dir),
        "cm_frame": cm_frame,
        "cm_count": cm_count,
        "size": plot_recall_by_size(runs, scores, out_dir),
        "eyes": plot_eyes(runs, stability, out_dir),
        "errors": plot_error_examples(runs, scores, truth, video, out_dir),
    }
    write_csv(runs, out_dir)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "meta": meta, "latency": stats, "accuracy": scores, "stability": stability,
            "settings": {n: r["settings"] for n, r in runs.items()},
        }, f, indent=2, default=str)
    write_summary(out_dir, meta, runs, stats, scores, stability, images, args.iou)

    print("\nReport written to", out_dir)
    for n in names:
        s, a = stats[n], scores[n]
        print("  %-4s %.1f ms mean (p95 %.1f) = %.1f fps | precision %.3f recall %.3f F1 %.3f | TP %d FP %d FN %d" % (
            n, s["mean_ms"], s["p95_ms"], s["fps_mean"], a["precision"], a["recall"], a["f1"], a["tp"], a["fp"], a["fn"]))


if __name__ == "__main__":
    main()
