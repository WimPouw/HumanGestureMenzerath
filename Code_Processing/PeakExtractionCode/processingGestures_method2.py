"""
Gesture Submovement Analysis (peak method 2: relative speed threshold)
Processes NPZ skeleton files (mediapipe world landmarks) to extract submovement metrics.
Peaks are local maxima in index finger speed above 10% of the max speed of that gesture, at least 160 ms apart.
A submovement is the interval between two consecutive peaks.
"""

import os
import hashlib
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from tqdm import tqdm

# landmark indices in 23-landmark upper body array
LM = {
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_index": 19, "right_index": 20
}

# parameters
SIGMA = 2                 # gaussian sd in frames (2 frames = 80 ms at 25 fps)
PEAK_DIST = 4             # min distance between peaks in frames (4 frames = 160 ms at 25 fps)
FPS = 25
PEAK_HEIGHT_RATIO = 0.10  # 10% of max speed
MOVE_THRESH = 0.15        # m/s (15 cm/s), only used for handedness (same as method 1)


def get_xyz(world, idx):
    # world shape: (frames, 92), each landmark = 4 values (x,y,z,vis)
    return world[:, idx*4 : idx*4+3]


def get_speed(pos, ts):
    # smooth position per axis and compute 3D speed (m/s)
    smooth = np.column_stack([gaussian_filter1d(pos[:,i], SIGMA) for i in range(3)])
    dt = np.diff(ts)
    dt[dt == 0] = 1/FPS  # guard against duplicate timestamps
    vel = np.diff(smooth, axis=0) / dt[:, None]
    speed = np.sqrt((vel**2).sum(axis=1))
    # diff loses one frame, so repeat the first speed value
    # (padding with 0 gives a fake peak when the hand already moves at clip start)
    speed = np.concatenate([[speed[0]], speed])
    # smooth the speed once more
    return gaussian_filter1d(speed, SIGMA)


def get_path_length(pos):
    # total distance travelled (m) of the smoothed position
    smooth = np.column_stack([gaussian_filter1d(pos[:,i], SIGMA) for i in range(3)])
    return np.sqrt((np.diff(smooth, axis=0)**2).sum(axis=1)).sum()


def parse_filename(fpath):
    # parse: CORPUS_speaker_clipid_gesture_subtype_skeleton.npz
    fname = os.path.basename(fpath).replace("_skeleton.npz", "")
    parts = fname.split("_")

    # find gesture index
    gidx = next((i for i,p in enumerate(parts) if p.lower() == "gesture"), None)
    if gidx is None or gidx < 2:
        return None

    # clip id sits right before "gesture", speaker is everything between corpus and clip id
    corpus = parts[0]
    clip_id = parts[gidx - 1]
    speaker = "_".join(parts[1:gidx-1]) if gidx > 2 else parts[1]
    subtype = parts[gidx + 1] if gidx + 1 < len(parts) else "NA"

    # ZHUBO single speaker
    if corpus == "ZHUBO":
        speaker = "zhubo"

    return {"corpus": corpus, "speaker": speaker, "clip_id": clip_id, "subtype": subtype}


def process_file(fpath):
    # parse metadata
    meta = parse_filename(fpath)
    if meta is None:
        return None, None, None

    # condition from folder
    condition = "objman" if "objman" in fpath.lower() else "gesture"

    # load npz
    data = np.load(fpath, allow_pickle=True)

    # find world landmarks array (shape n x 92)
    world = None
    for k in data.files:
        if isinstance(data[k], np.ndarray) and data[k].ndim == 2 and data[k].shape[1] == 92:
            world = data[k]
            break
    if world is None or len(world) < 3:
        return None, None, None

    # fingerprint of the skeleton data, used to find files with exactly the same data
    data_key = hashlib.md5(np.ascontiguousarray(np.round(world, 6) + 0.0).tobytes()).hexdigest()
    
    # timestamps (in seconds), fall back on frame number / fps
    ts = data["timestamps"] if "timestamps" in data.files else np.arange(len(world)) / FPS

    # extract positions
    left_idx = get_xyz(world, LM["left_index"])
    right_idx = get_xyz(world, LM["right_index"])

    # path lengths
    left_path = get_path_length(left_idx)
    right_path = get_path_length(right_idx)

    # handedness based on whether the absolute movement threshold is reached
    # (10% of the max over both hands is reached by almost any hand, so everything would be bimanual)
    left_speed = get_speed(left_idx, ts)
    right_speed = get_speed(right_idx, ts)
    left_moves = left_speed.max() >= MOVE_THRESH
    right_moves = right_speed.max() >= MOVE_THRESH

    if left_moves and right_moves:
        hand = "bimanual"
    elif left_moves:
        hand = "left"
    elif right_moves:
        hand = "right"
    else:
        hand = "none"

    # dominant side = the hand with the longest path, all further analysis is on this side
    if left_path >= right_path:
        dom = "left"
        idx_pos = left_idx
        wrist_pos = get_xyz(world, LM["left_wrist"])
        elbow_pos = get_xyz(world, LM["left_elbow"])
        shoulder_pos = get_xyz(world, LM["left_shoulder"])
    else:
        dom = "right"
        idx_pos = right_idx
        wrist_pos = get_xyz(world, LM["right_wrist"])
        elbow_pos = get_xyz(world, LM["right_elbow"])
        shoulder_pos = get_xyz(world, LM["right_shoulder"])

    # speeds
    speed_idx = get_speed(idx_pos, ts)
    speed_wrist = get_speed(wrist_pos, ts)
    speed_elbow = get_speed(elbow_pos, ts)
    speed_shoulder = get_speed(shoulder_pos, ts)

    # peak threshold is 10% of max speed
    # note: there is no absolute floor here, so tracking jitter of a still hand also gives peaks
    # (max_speed is saved, so these can be filtered in the analysis)
    peak_thresh = speed_idx.max() * PEAK_HEIGHT_RATIO

    # peaks = local maxima above the relative threshold and at least PEAK_DIST frames apart
    # index finger peaks define the submovements, the other joints are only counted (each with its own max)
    peaks_idx, _ = find_peaks(speed_idx, distance=PEAK_DIST, height=peak_thresh)
    peaks_wrist, _ = find_peaks(speed_wrist, distance=PEAK_DIST, height=speed_wrist.max() * PEAK_HEIGHT_RATIO)
    peaks_elbow, _ = find_peaks(speed_elbow, distance=PEAK_DIST, height=speed_elbow.max() * PEAK_HEIGHT_RATIO)
    peaks_shoulder, _ = find_peaks(speed_shoulder, distance=PEAK_DIST, height=speed_shoulder.max() * PEAK_HEIGHT_RATIO)

    # metrics
    dur_ms = (ts[-1] - ts[0]) * 1000
    n_peaks = len(peaks_idx)
    # a submovement is the interval between two peaks, so number of intervals is one less than number of peaks
    n_subm = max(n_peaks - 1, 0)
    # single unit = fewer than two peaks, so there is no interval (these are excluded in the analysis)
    single_unit = 1 if n_peaks <= 1 else 0

    if n_peaks > 1:
        peak_times = np.sort(ts[peaks_idx])
        intervals = np.diff(peak_times) * 1000
        mean_subm_dur = intervals.mean()
    else:
        # no interval, so we fall back on the whole gesture duration
        mean_subm_dur = dur_ms

    # id used as random intercept in the models
    # we use the file name, as corpus_speaker_clipid is not unique in MULTISIMO and SAGA
    gid = os.path.basename(fpath).replace("_skeleton.npz", "")

    # flat row (one row per gesture)
    flat = {
        "gesture_id": gid,
        "corpus": meta["corpus"],
        "speaker": meta["speaker"],
        "clip_id": meta["clip_id"],
        "subtype": meta["subtype"],
        "condition": condition,
        "duration_ms": dur_ms,
        "mean_subm_dur_ms": mean_subm_dur,
        "n_submovements": n_subm,
        "single_unit": single_unit,
        "handedness": hand,
        "dominant_side": dom,
        "path_left": left_path,
        "path_right": right_path,
        "path_dominant": get_path_length(idx_pos),
        "max_speed": speed_idx.max(),
        "n_peaks_index": len(peaks_idx),
        "n_peaks_wrist": len(peaks_wrist),
        "n_peaks_elbow": len(peaks_elbow),
        "n_peaks_shoulder": len(peaks_shoulder)
    }

    # long rows - base info shared across submovements (same n_submovements as in flat)
    base = {
        "gesture_id": gid,
        "corpus": meta["corpus"],
        "speaker": meta["speaker"],
        "clip_id": meta["clip_id"],
        "subtype": meta["subtype"],
        "condition": condition,
        "duration_ms": dur_ms,
        "n_submovements": n_subm,
        "single_unit": single_unit,
        "handedness": hand,
        "dominant_side": dom,
        "path_left": left_path,
        "path_right": right_path,
        "path_dominant": get_path_length(idx_pos),
        "max_speed": speed_idx.max(),
        "n_peaks_index": len(peaks_idx),
        "n_peaks_wrist": len(peaks_wrist),
        "n_peaks_elbow": len(peaks_elbow),
        "n_peaks_shoulder": len(peaks_shoulder)
    }

    long_rows = []
    if n_peaks > 1:
        # one row per submovement, running from peak i to peak i+1
        peak_times = np.sort(ts[peaks_idx])
        for i in range(len(peak_times) - 1):
            row = base.copy()
            row.update({
                "subm_index": i + 1,
                "subm_dur_ms": (peak_times[i+1] - peak_times[i]) * 1000,
                "peak_time_ms": peak_times[i] * 1000,
                "peak_speed": speed_idx[peaks_idx[i]]
            })
            long_rows.append(row)
    else:
        # single unit: one placeholder row with the whole gesture duration (n_submovements = 0)
        row = base.copy()
        row.update({
            "subm_index": 1,
            "subm_dur_ms": dur_ms,
            "peak_time_ms": ts[peaks_idx[0]] * 1000 if len(peaks_idx) > 0 else dur_ms / 2,
            "peak_speed": speed_idx.max()
        })
        long_rows.append(row)

    return flat, long_rows, data_key


# main
input_dirs = ["../../Data_Input/Gesture", "../../Data_Input/Objman"]
output_dir = "../../Data_Output"

# collect files
npz_files = []
for d in input_dirs:
    if os.path.exists(d):
        for root, _, files in os.walk(d):
            npz_files += [os.path.join(root, f) for f in files if f.endswith("_skeleton.npz")]

print(f"Found {len(npz_files)} files")

# process
flat_rows = []
long_rows = []
seen = {}     # fingerprint of the skeleton data -> file that we kept
removed = []  # duplicate files that we removed

for fpath in tqdm(sorted(npz_files)):
    flat, longs, data_key = process_file(fpath)
    if flat:
        # the same event can be exported twice under two labels (e.g., SAGA_12_19 as deictic and as iconic)
        # we keep the first file, and remove later files that have exactly the same skeleton data
        if data_key in seen:
            removed.append({"removed": os.path.basename(fpath), "kept": seen[data_key]})
            continue
        seen[data_key] = os.path.basename(fpath)
        flat_rows.append(flat)
        long_rows.extend(longs)

# save (flat = one row per gesture, long = one row per submovement)
os.makedirs(output_dir, exist_ok=True)
df_flat = pd.DataFrame(flat_rows)
df_long = pd.DataFrame(long_rows)

df_flat.to_csv(f"{output_dir}/gestures_flat_method2.csv", index=False)
df_long.to_csv(f"{output_dir}/gestures_long_method2.csv", index=False)

# save which duplicates were removed
df_removed = pd.DataFrame(removed, columns=["removed", "kept"])
df_removed.to_csv(f"{output_dir}/duplicates_removed_method2.csv", index=False)
print(f"\nDuplicates removed (same skeleton data): {len(df_removed)}")
if len(df_removed) > 0:
    print(df_removed["removed"].str.split("_").str[0].value_counts().to_string())

print(f"\nFlat: {len(df_flat)} gestures")
print(f"Long: {len(df_long)} submovements")

if len(df_flat) > 0:
    print(f"\nCorpus counts:\n{df_flat['corpus'].value_counts()}")
    print(f"\nCondition counts:\n{df_flat['condition'].value_counts()}")
    print(f"\nSingle units (excluded in analysis): {df_flat['single_unit'].sum()}")
    
    # numbers for reporting: speakers and gestures per corpus (after removing duplicates)
    counts = df_flat.groupby(["condition", "corpus"]).agg(
        n_speakers=("speaker", "nunique"),
        n_gestures=("gesture_id", "count"),
        n_in_analysis=("single_unit", lambda s: int((s == 0).sum())))
    print("\nSpeakers and gestures per corpus:")
    for (cond, corp), r in counts.iterrows():
        print(f"  n speakers = {r['n_speakers']}, n gestures = {r['n_gestures']} ({r['n_in_analysis']} in analysis), {corp} [{cond}]")
    for cond, r in counts.groupby(level="condition").sum().iterrows():
        print(f"  total [{cond}]: n gestures = {r['n_gestures']} ({r['n_in_analysis']} in analysis)")

    # gesture ids are the random intercepts, so they should be unique per event
    n_dup = df_flat["gesture_id"].duplicated().sum()
    if n_dup > 0:
        print(f"\nWarning: {n_dup} duplicated gesture ids")
