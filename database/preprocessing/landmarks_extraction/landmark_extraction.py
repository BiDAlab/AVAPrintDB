import os
import cv2
import argparse
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Same landmark IDs used in https://doi.org/10.1109/IJCB65343.2025.11411089
LANDMARK_IDS = [
        46,53,52,65,55,285,295,282,283,276,70,63,105,66,107,
        336,296,334,293,300,33,161,160,159,158,157,
        133,154,153,145,144,468,362,384,385,
        386,387,388,263,373,374,380,381,473,
        205,50,425,64,294,
        280,9,168,5,4,19,185,40,39,37,
        0,267,269,270,409,191,80,82,13,312,310,415,95,
        88,178,87,14,317,402,318,324,146,91,181,84,17,314,
        405,321,375,61,291,10,297,284,389,454,
        361,397,379,400,152,
        176,150,172,132,234,162,54,
        67
    ]

def npy_to_viz_mp4(
    npy_path: str,
    mp4_out_path: str,
    out_h: int = 512,
    out_w: int = 512,
    fps: float = 30.0,
    bg_color=(0, 0, 0),
    pt_color=(255, 255, 255),
    radius: int = 2,
):
    """
    Load a single .npy containing normalized landmarks and generate an MP4 visualization.

    Expected .npy format:
      - np.ndarray of shape (T, N, 3) where T=#frames, N=#landmarks
      - values are normalized (nose-centered, inter-ocular scaled)
      - may contain NaNs

    Output:
      - MP4 (one panel) showing the rendered normalized landmarks per frame.

    Notes:
      - If fps is unknown, default is 30.0.
      - If the .npy is empty or invalid, raises a ValueError.
    """
    arr = np.load(npy_path)

    if not isinstance(arr, np.ndarray):
        raise ValueError(f"{npy_path} did not contain a numpy array.")

    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(
            f"Expected array shape (T, N, 3). Got {arr.shape} from {npy_path}."
        )

    T = arr.shape[0]
    if T == 0:
        raise ValueError(f"No frames found in {npy_path} (T=0).")

    ensure_parent_dir(mp4_out_path)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(mp4_out_path, fourcc, float(fps), (int(out_w), int(out_h)), True)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for: {mp4_out_path}")

    try:
        for t in range(T):
            frame = render_normalized_landmarks(
                norm_pts_xyz=arr[t],
                out_h=int(out_h),
                out_w=int(out_w),
                bg_color=bg_color,
                pt_color=pt_color,
                radius=radius,
            )
            writer.write(frame)
    finally:
        writer.release()



def compute_recommended_px_per_unit(
    arrays,
    out_w,
    out_h,
    margin=0.10,
    percentile=99.0,
):
    """
    Computes a single fixed px_per_unit from the data distribution (NOT per-frame).
    Uses a robust percentile of |x| and |y| to ignore outliers.

    arrays: list of np.ndarray with shape (T,N,3)
    """
    xs = []
    ys = []
    for arr in arrays:
        xy = arr[..., :2].reshape(-1, 2)
        finite = np.isfinite(xy).all(axis=1)
        xy = xy[finite]
        if xy.size == 0:
            continue
        xs.append(np.abs(xy[:, 0]))
        ys.append(np.abs(xy[:, 1]))

    if not xs:
        raise ValueError("No finite points found in arrays.")

    ax = np.concatenate(xs)
    ay = np.concatenate(ys)

    x_ref = np.percentile(ax, percentile)
    y_ref = np.percentile(ay, percentile)

    usable_w = out_w * (1.0 - 2 * margin)
    usable_h = out_h * (1.0 - 2 * margin)

    # We want x_ref units to map to half usable width, similarly for y.
    # px_per_unit must satisfy:
    #   x_ref * px_per_unit <= usable_w/2
    #   y_ref * px_per_unit <= usable_h/2
    # so px_per_unit <= min(usable_w/(2*x_ref), usable_h/(2*y_ref))
    # (guard against zeros)
    x_ref = float(max(x_ref, 1e-8))
    y_ref = float(max(y_ref, 1e-8))

    return min(usable_w / (2.0 * x_ref), usable_h / (2.0 * y_ref))


def draw_landmarks_nose_space(
    canvas,
    pts_xyz,
    px_per_unit,
    color,
    radius=2,
    center=None,
):
    """
    pts_xyz: (N,3) in nose-centered, interocular=1 coordinate system
    Draws using a fixed mapping: pixel = center + xy * px_per_unit.
    """
    if pts_xyz is None:
        return

    xy = pts_xyz[:, :2]
    finite = np.isfinite(xy).all(axis=1)
    xy = xy[finite]
    if xy.size == 0:
        return

    h, w = canvas.shape[:2]
    if center is None:
        cx, cy = w / 2.0, h / 2.0
    else:
        cx, cy = center

    # Convert to pixel coordinates
    px = np.rint(cx + xy[:, 0] * px_per_unit).astype(np.int32)
    py = np.rint(cy + xy[:, 1] * px_per_unit).astype(np.int32)  # keep y-down convention

    for x, y in zip(px, py):
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(canvas, (int(x), int(y)), int(radius), color, thickness=-1)


def npy_pair_to_overlay_mp4_fixedscale(
    npy_a_path: str,
    npy_b_path: str,
    mp4_out_path: str,
    out_h: int = 512,
    out_w: int = 512,
    fps: float = 30.0,
    bg_color=(0, 0, 0),
    color_a=(255, 255, 255),
    color_b=(0, 255, 0),
    radius: int = 2,
    px_per_unit =None,
    center=None,  # (cx, cy) in pixels; default is canvas center
    recommend_from_data: bool = True,
    recommend_percentile: float = 99.0,
    margin: float = 0.10,
):
    """
    Overlay two .npy landmark sequences in nose-centered/interocular=1 space onto one MP4.

    - Different lengths: writes max(Ta, Tb) frames.
    - Fixed mapping: px = cx + x*px_per_unit; py = cy + y*px_per_unit
    - No per-frame scaling/fitting is used.
    - If px_per_unit is None and recommend_from_data is True, compute a single fixed
      px_per_unit using a robust percentile over BOTH arrays.
    """
    A = np.load(npy_a_path)
    B = np.load(npy_b_path)

    def check(arr, name):
        if not isinstance(arr, np.ndarray) or arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"{name} must have shape (T, N, 3). Got {getattr(arr, 'shape', None)}")

    check(A, "npy_a")
    check(B, "npy_b")

    Ta, Na, _ = A.shape
    Tb, Nb, _ = B.shape
    if Na != Nb:
        raise ValueError(f"Different number of landmarks: A has {Na}, B has {Nb}.")

    T = max(Ta, Tb)
    if T == 0:
        raise ValueError("Both arrays have zero frames.")

    if px_per_unit is None:
        if not recommend_from_data:
            raise ValueError("px_per_unit is None. Provide px_per_unit or set recommend_from_data=True.")
        px_per_unit = compute_recommended_px_per_unit(
            arrays=[A, B],
            out_w=out_w,
            out_h=out_h,
            margin=margin,
            percentile=recommend_percentile,
        )
        print(f"[viz] recommended px_per_unit={px_per_unit:.3f} (percentile={recommend_percentile}, margin={margin})")

    ensure_parent_dir(mp4_out_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(mp4_out_path, fourcc, float(fps), (int(out_w), int(out_h)), True)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for: {mp4_out_path}")

    try:
        for t in range(T):
            canvas = np.zeros((int(out_h), int(out_w), 3), dtype=np.uint8)
            if bg_color != (0, 0, 0):
                canvas[:] = bg_color

            pts_a = A[t] if t < Ta else None
            pts_b = B[t] if t < Tb else None

            draw_landmarks_nose_space(canvas, pts_a, px_per_unit, color_a, radius=radius, center=center)
            draw_landmarks_nose_space(canvas, pts_b, px_per_unit, color_b, radius=radius, center=center)

            writer.write(canvas)
    finally:
        writer.release()



def normalize_landmarks(
    landmarks,
    nose_index,
    left_eye_index,
    right_eye_index,
    last_valid_anchors=None,
    warn_prefix="",
):
    """
    Normalize landmarks by:
      - translating so nose is origin
      - scaling by inter-ocular distance (left eye <-> right eye)

    If nose or eyes are NaN in the current frame, use previous-frame anchor coords.
    Prints a warning for every frame where any of the anchors are NaN.

    Returns:
      (norm_landmarks or None, updated_last_valid_anchors)
    """
    if last_valid_anchors is None:
        last_valid_anchors = {"nose": None, "left": None, "right": None}

    nose = landmarks[nose_index].copy()
    left = landmarks[left_eye_index].copy()
    right = landmarks[right_eye_index].copy()

    nose_bad = np.any(np.isnan(nose))
    left_bad = np.any(np.isnan(left))
    right_bad = np.any(np.isnan(right))

    if nose_bad or left_bad or right_bad:
        print(
            f"WARNING {warn_prefix} anchor NaN(s): "
            f"nose={nose_bad}, left_eye={left_bad}, right_eye={right_bad} "
            f"-> using previous frame anchors when available",
            flush=True,
        )

    # Fallback to previous anchors if needed
    if nose_bad:
        if last_valid_anchors["nose"] is None:
            return None, last_valid_anchors
        nose = last_valid_anchors["nose"]
    if left_bad:
        if last_valid_anchors["left"] is None:
            return None, last_valid_anchors
        left = last_valid_anchors["left"]
    if right_bad:
        if last_valid_anchors["right"] is None:
            return None, last_valid_anchors
        right = last_valid_anchors["right"]

    # Update history only from real detections (not fallbacks)
    if not nose_bad:
        last_valid_anchors["nose"] = nose
    if not left_bad:
        last_valid_anchors["left"] = left
    if not right_bad:
        last_valid_anchors["right"] = right

    centered = landmarks - nose
    dist = np.linalg.norm((left - nose) - (right - nose))

    if not np.isfinite(dist) or dist <= 0:
        print(f"WARNING {warn_prefix} invalid interocular distance ({dist}); skipping normalization", flush=True)
        return None, last_valid_anchors

    return centered / dist, last_valid_anchors


# ---------------- MediaPipe landmarker ---------------- #

def create_landmarker(model_path, prefer_gpu=True):
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    FaceLandmarker = mp.tasks.vision.FaceLandmarker

    if prefer_gpu:
        try:
            options = FaceLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=model_path,
                    delegate=BaseOptions.Delegate.GPU,
                ),
                running_mode=VisionRunningMode.VIDEO,
            )
            return FaceLandmarker.create_from_options(options)
        except Exception as e:
            print(f"WARNING: GPU failed, falling back to CPU: {e}", flush=True)

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=model_path,
            delegate=BaseOptions.Delegate.CPU,
        ),
        running_mode=VisionRunningMode.VIDEO,
    )
    return FaceLandmarker.create_from_options(options)


# ---------------- Visualization helpers ---------------- #

def draw_landmarks_overlay(bgr_frame, pts_xy_norm, color=(0, 255, 0), radius=2):
    """
    bgr_frame: HxWx3
    pts_xy_norm: (N,2) normalized coords in [0..1] from MediaPipe, may contain NaN
    """
    h, w = bgr_frame.shape[:2]
    out = bgr_frame.copy()
    if pts_xy_norm is None:
        return out

    for x, y in pts_xy_norm:
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        px = int(round(x * w))
        py = int(round(y * h))
        if 0 <= px < w and 0 <= py < h:
            cv2.circle(out, (px, py), radius, color, thickness=-1)
    return out


def render_normalized_landmarks(
    norm_pts_xyz,
    out_h,
    out_w,
    bg_color=(0, 0, 0),
    pt_color=(255, 255, 255),
    radius=2,
):
    """
    norm_pts_xyz: (N,3) normalized landmarks (nose-centered, interocular-scaled), may contain NaN
    Renders to a blank canvas.

    IMPORTANT: MediaPipe landmark coords use image convention:
      x increases to the right, y increases downward.
    So we should NOT flip y when plotting, otherwise the result appears upside down.
    """
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    if bg_color != (0, 0, 0):
        canvas[:] = bg_color

    if norm_pts_xyz is None:
        return canvas

    xy = norm_pts_xyz[:, :2]
    finite = np.isfinite(xy).all(axis=1)
    xy = xy[finite]
    if xy.size == 0:
        return canvas

    # Per-frame scaling: fit all points
    max_abs = float(np.max(np.abs(xy)))
    if not np.isfinite(max_abs) or max_abs <= 0:
        return canvas

    margin = 0.10
    usable_w = out_w * (1.0 - 2 * margin)
    usable_h = out_h * (1.0 - 2 * margin)
    scale = min(usable_w, usable_h) / (2.0 * max_abs)

    cx = out_w / 2.0
    cy = out_h / 2.0

    # DO NOT flip y: keep MediaPipe's convention (y downward)
    for x, y in xy:
        px = int(round(cx + x * scale))
        py = int(round(cy + y * scale))
        if 0 <= px < out_w and 0 <= py < out_h:
            cv2.circle(canvas, (px, py), radius, pt_color, thickness=-1)

    return canvas


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


# ---------------- Database filtering ---------------- #

def database_allows_filename(database: str, filename: str) -> bool:
    """
    filename: base filename (e.g., "Actor_01.mp4"), not a full path.
    Matching is case-sensitive
    """
    if database == "CREMAD":
        return filename.startswith("C")
    if database == "RAVDESS":
        return filename.startswith("Actor")
    raise ValueError(f"Unsupported database: {database}")


# ---------------- Video processing ---------------- #

def process_video(
    path_in,
    path_out_npy,
    path_out_viz,
    log_path,
    landmark_ids,
    model_path,
    prefer_gpu=True,
    gpu_id=None,
    visualize=False,
):
    """
    Produces:
      - .npy of normalized landmarks per frame (only frames where normalization succeeds)
      - (optional) a single .mp4 that contains 3 side-by-side panels:
          [original | original+landmarks(pre-normalization) | normalized-landmarks-on-blank]
    """
    try:
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        landmarker = create_landmarker(model_path, prefer_gpu)

        cap = cv2.VideoCapture(path_in)
        if not cap.isOpened():
            with open(log_path, "a") as f:
                f.write(f"OPEN_ERROR: {path_in}\n")
            landmarker.close()
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0

        # Required IDs for anchors must exist in the selected list
        nose_i = landmark_ids.index(4)
        left_i = landmark_ids.index(133)
        right_i = landmark_ids.index(362)

        writer = None
        base_w = None
        base_h = None

        all_norm_frames = []
        last_valid_anchors = {"nose": None, "left": None, "right": None}
        frame_idx = 0

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if base_w is None:
                base_h, base_w = frame_bgr.shape[:2]
                if visualize:
                    ensure_parent_dir(path_out_viz)
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(path_out_viz, fourcc, fps, (base_w * 3, base_h), True)
                    if not writer.isOpened():
                        with open(log_path, "a") as f:
                            f.write(f"VIZ_WRITER_OPEN_ERROR: {path_out_viz}\n")
                        writer = None  # proceed without visualization

            img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

            # VIDEO mode wants monotonic timestamps
            timestamp_ms = int((frame_idx / fps) * 1000)
            res = landmarker.detect_for_video(mp_img, timestamp_ms)
            warn_prefix = f"[{os.path.basename(path_in)} frame={frame_idx} ts={timestamp_ms}ms gpu_id={gpu_id}]"
            frame_idx += 1

            pts = None
            pts_xy = None
            norm = None

            if res.face_landmarks:
                lm = res.face_landmarks[0]
                pts = np.full((len(landmark_ids), 3), np.nan, dtype=np.float32)
                for i, lid in enumerate(landmark_ids):
                    if lid < len(lm):
                        p = lm[lid]
                        pts[i] = (p.x, p.y, p.z)

                pts_xy = pts[:, :2]  # pre-normalization coordinates (x,y) in [0..1]
                norm, last_valid_anchors = normalize_landmarks(
                    pts, nose_i, left_i, right_i,
                    last_valid_anchors=last_valid_anchors,
                    warn_prefix=warn_prefix,
                )
                if norm is not None:
                    all_norm_frames.append(norm)

            # Visualization: write *every* frame, even if no landmarks
            if visualize and writer is not None and base_w is not None:
                panel_orig = frame_bgr
                panel_overlay = draw_landmarks_overlay(frame_bgr, pts_xy, color=(0, 255, 0), radius=2)
                panel_norm = render_normalized_landmarks(norm, base_h, base_w, bg_color=(0, 0, 0), pt_color=(255, 255, 255), radius=2)

                combined = np.concatenate([panel_orig, panel_overlay, panel_norm], axis=1)
                writer.write(combined)

        cap.release()
        if writer is not None:
            writer.release()
        landmarker.close()

        if not all_norm_frames:
            with open(log_path, "a") as f:
                f.write(f"NO_USABLE_FACE: {path_in}\n")
            return

        ensure_parent_dir(path_out_npy)
        np.save(path_out_npy, np.stack(all_norm_frames, axis=0))

    except Exception as e:
        with open(log_path, "a") as f:
            f.write(f"ERROR ({path_in}): {str(e)}\n")


# ---------------- Main orchestration ---------------- #

def parse_gpus(gpu_str):
    if not gpu_str:
        return None
    gpu_str = gpu_str.strip()
    if not gpu_str:
        return None
    return [int(x.strip()) for x in gpu_str.split(",") if x.strip()]


def main(
    input_folder,
    output_folder,
    landmark_ids,
    model_path,
    database,
    max_workers=None,
    prefer_gpu=True,
    gpus=None,
    visualize=False,
    regenerate=False,
):
    """
    Mirrors directory structure:
      out_path_npy = output_folder / rel_dir_from_input / <video_stem>.npy
      out_path_viz = output_folder / rel_dir_from_input / <video_stem>__viz.mp4   (if visualize)
    """
    if max_workers is None:
        max_workers = len(gpus) if (prefer_gpu and gpus) else multiprocessing.cpu_count()

    os.makedirs(output_folder, exist_ok=True)
    log_path = os.path.join(output_folder, "error_videos.txt")
    open(log_path, "w").close()

    jobs = []
    skipped = 0

    for root, _, files in os.walk(input_folder):
        for fn in files:
            if not fn.lower().endswith(".mp4"):
                continue

            if not database_allows_filename(database, fn):
                continue

            in_path = os.path.join(root, fn)

            rel_dir = os.path.relpath(root, input_folder)
            out_dir = os.path.join(output_folder, rel_dir)

            stem, _ = os.path.splitext(fn)
            out_npy = os.path.join(out_dir, stem + ".npy")
            out_viz = os.path.join(out_dir, stem + "__viz.mp4") if visualize else None

            if not regenerate:
                npy_exists = os.path.exists(out_npy)
                viz_exists = (not visualize) or (out_viz is not None and os.path.exists(out_viz))

                # Skip if outputs we care about already exist
                if npy_exists and viz_exists:
                    skipped += 1
                    continue

            jobs.append((in_path, out_npy, out_viz))

    if not jobs:
        print(f"No videos to process. Skipped {skipped} already-generated outputs.", flush=True)
        return

    if skipped:
        print(f"Skipping {skipped} videos (outputs already present). Use --regenerate to force.", flush=True)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, (in_path, out_npy, out_viz) in enumerate(jobs):
            gpu_id = gpus[i % len(gpus)] if (prefer_gpu and gpus) else None
            futures.append(
                executor.submit(
                    process_video,
                    in_path,
                    out_npy,
                    out_viz,
                    log_path,
                    landmark_ids,
                    model_path,
                    prefer_gpu,
                    gpu_id,
                    visualize,
                )
            )

        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing videos"):
            try:
                f.result()
            except Exception as e:
                with open(log_path, "a") as log:
                    log.write(f"FUTURE_ERROR: {str(e)}\n")


# ---------------- Entry point ---------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and normalize facial landmarks from videos.")
    parser.add_argument('--input_folder', type=str, default="AVAPrintDB_v0/database/data/videos/TEST/LIVE", help='Path to the input folder containing videos.')
    parser.add_argument('--output_folder', type=str, default="/tmp", help='Path to the output folder to save landmarks.')
    parser.add_argument('--max_workers', type=int, default=1, help='Maximum number of parallel workers.')
    parser.add_argument("--model_path", type=str, default="AVAPrintDB_v0/database/preprocessing/landmarks_extraction/face_landmarker.task", help="Path to the MediaPipe face landmarker model.")
    parser.add_argument("--prefer_gpu", action="store_true")
    parser.add_argument("--gpus", type=str, default=None, help='Comma-separated GPU IDs, e.g. "0,1"')
    parser.add_argument(
        "--database",
        type=str,
        required=True,
        choices=["CREMAD", "RAVDESS"],
        help='Which database naming convention to use for filtering input videos.',
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Also write a single MP4 with 3 panels: original | original+landmarks | normalized-landmarks-on-blank",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Force regeneration even if output files already exist.",
    )

    args = parser.parse_args()
    gpus = parse_gpus(args.gpus)

    main(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        landmark_ids=LANDMARK_IDS,
        model_path=args.model_path,
        database=args.database,
        max_workers=args.max_workers,
        prefer_gpu=args.prefer_gpu,
        gpus=gpus,
        visualize=args.visualize,
        regenerate=args.regenerate,
    )
