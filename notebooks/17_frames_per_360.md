---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.3
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter1d, median_filter
from tqdm.auto import tqdm
```

```python
intervals = pd.read_csv("/Users/gimli/cvr/data/zavity/ETE 2026_03_16-all/out/GX011155-breakpoints.csv")
angles = pd.read_csv("/Users/gimli/cvr/data/zavity/ETE 2026_03_16-all/out/GX011155-full_angles.csv")
motions = np.load("/Users/gimli/cvr/data/zavity/ETE 2026_03_16-all/out/GX011155-preprocessed-motion_local_diff.npy")
features = np.load("/Users/gimli/cvr/data/zavity/ETE 2026_03_16-all/out/GX011155-preprocessed-features_motion.npy")
```

```python
rot_intervals = intervals[intervals["sequence type (rot 1, shift 0)"]==1]
shift_intervals = intervals[intervals["sequence type (rot 1, shift 0)"]==0]
```

```python
plt.figure(figsize=(15, 4))
plt.plot(angles["angle (deg)"])
for rowid, interval in intervals.iterrows():
    if interval["sequence type (rot 1, shift 0)"] == 1:
        plt.axvspan(interval["sequence start"], interval["sequence end"], alpha=0.2, color="red")
plt.show()
```

```python
plt.figure(figsize=(15, 4))
plt.plot(median_filter(np.linalg.norm(motions[:,1:], axis=1), 21))
for rowid, interval in rot_intervals.iterrows():
    plt.axvspan(interval["sequence start"], interval["sequence end"], alpha=0.2, color="red")
plt.show()
```

```python
start_to_interval = np.where(np.array([np.logical_and(features[:, 0] >= interval["sequence start"], features[:, 0] < interval["sequence end"]) for rid, interval in intervals.iterrows()]).T)
end_to_interval = np.where(np.array([np.logical_and(features[:, 1] >= interval["sequence start"], features[:, 1] < interval["sequence end"]) for rid, interval in intervals.iterrows()]).T)
```

```python
start_color = np.ones((features.shape[0])) * -1
start_color[feature_to_interval[0]] = start_to_interval[1]
end_color = np.ones((features.shape[0])) * -1
end_color[end_to_interval[0]] = end_to_interval[1]
```

```python
intervals
```

```python
plt.figure(figsize=(15, 3))
ax = plt.subplot(111)
ax.scatter(np.cumsum(features[:, 2] * features[:, 4]),
           np.cumsum(features[:, 3] * features[:, 4]), c=start_color, marker="+", cmap="Set3", alpha=0.1)
plt.title("Global motion features, start of interval")
plt.show()

plt.figure(figsize=(15, 3))
ax = plt.subplot(111)
ax.scatter(np.cumsum(features[:, 2] * features[:, 4]),
           np.cumsum(features[:, 3] * features[:, 4]), c=end_color, marker="+", cmap="Set3")
plt.title("Global motion features, end of interval")
plt.show()
```

```python
xx, yy = [], []
for feature in features:
    xx.append(feature[0])
    yy.append(feature[4] / (feature[1] - feature[0]))
    xx.append(feature[1]-1)
    yy.append(feature[4] / (feature[1] - feature[0]))
```

```python
plt.figure(figsize=(15, 3))
plt.scatter(features[:, 1] - features[:, 0], features[:, 4]/(features[:, 1] - features[:, 0]), alpha=0.5)
plt.ylim(0,20)
plt.show()
```

```python
motion_from_global = np.zeros((len(motions), 3))
motion_from_global[:, 0] = motions[:, 0]
for iid, interval in intervals.iterrows():
    for fno in np.arange(interval["sequence start"], interval["sequence end"]):
        corresponding_feature = np.where(np.logical_and(features[:,0] <= fno, features[:, 1] > fno))
        if corresponding_feature[0].size > 0:
            fno_min, fno_max, dx, dy, medmag, meanmag, angle_deg = features[corresponding_feature[0][0]]
            if interval["sequence type (rot 1, shift 0)"] == 1:
                motion_from_global[fno, 1:] = [meanmag / (fno_max - fno_min), 0 / (fno_max - fno_min)]
            else:
                motion_from_global[fno, 1:] = [0 / (fno_max - fno_min), meanmag / (fno_max - fno_min)]
```

```python
plt.figure()
ax = plt.subplot(111)
ax.scatter(np.cumsum(motion_from_global[:, 1]),
           np.cumsum(motion_from_global[:, 2]), marker="+", alpha=0.003)
ax.scatter(np.cumsum(features[:, 2] * features[:, 4]),
           np.cumsum(features[:, 3] * features[:, 4]), c=end_color, marker="+", cmap="Accent")
ax.set_aspect("equal")
plt.show()
```

```python
íshift_mask = np.zeros((motions.shape[0]), dtype=bool)
pixel_shift_per_oio_slice = 0
for iid, row in shift_intervals.iterrows():
    start, end = row["sequence start"], row["sequence end"]
    #fno, dx, dy, med, mean, angle = motions[start:end]
    plt.figure(figsize=(15, 3))
    plt.plot(np.arange(start, end), motions[start:end, -2])
    #plt.plot(np.arange(start, end), median_filter(motions[start:end, 3], 1))
    #plt.plot(xx, yy)
    #plt.plot(motions[start:end, -2] * motions[start:end, 3])
    y_loc = np.cumsum(motions[start:end, 2] * motions[start:end, 3])
    pixel_shift_per_oio_slice += np.max(y_loc) - np.min(y_loc)
    print(f"Total horizontal movement: {np.max(y_loc) - np.min(y_loc):.2f}")
    for feature in features:
        if start <= feature[0] < end:
            plt.axvline(feature[0], color="red", alpha=0.5, linestyle="--")
    plt.axhline(-90, color="red")
    plt.show()
    # plt.figure()
    # ax = plt.subplot(111)
    # ax.scatter(np.cumsum(motions[start:end, 1] * motions[start:end, 3]),
    #            np.cumsum(motions[start:end, 2] * motions[start:end, 3]), c=motions[start:end, -1], marker="+", alpha=0.1)
    # ax.set_aspect("equal")
    # plt.show()
pixel_shift_per_oio_slice /= len(shift_intervals)
print(f"Average pixel shift per OIO slice: {pixel_shift_per_oio_slice:.2f}")
```

```python
start, end
```

```python
plt.plot(np.linalg.norm(motions[:,1:], axis=1)[shift_mask])
plt.show()
```

```python

image_width = (rot_intervals["sequence end"] - rot_intervals["sequence start"]) * speeds["horizontal"]
```

```python
speeds, image_width
```

```python
motions[rot_intervals["sequence start"][0], :], motions[rot_intervals["sequence end"][0]]
```

```python
len(angles)
```

```python
vidcap = cv2.VideoCapture("/Users/gimli/cvr/data/zavity/ETE 2026_03_16-all/out/GX011155-preprocessed.mp4")
```

```python
frames = []
success = True
frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
for fno in tqdm(range(frame_count), total=frame_count, desc="Reading frames"):
    success, frame = vidcap.read()
    if success:
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
```

```python
def flow_to_heatmap(mag, vmin=0.0, vmax=None):
    """
    Convert flow magnitude to colored heatmap (BGR).
    """
    if vmax is None:
        vmax = np.percentile(mag, 99) + 1e-6
    mag_clip = np.clip(mag, vmin, vmax)
    mag_norm = ((mag_clip - vmin) / (vmax - vmin + 1e-12) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(mag_norm, cv2.COLORMAP_JET)
    return heatmap


def draw_flow_arrows(
    image_bgr,
    flow,
    step=24,
    scale=1.0,
    min_magnitude=0.5,
    color=(255, 255, 255),
    thickness=1,
    tip_length=0.25,
):
    """
    Draw sparse arrows over image.
    """
    h, w = flow.shape[:2]

    ys = np.arange(step // 2, h, step)
    xs = np.arange(step // 2, w, step)

    for y in ys:
        for x in xs:
            dx, dy = flow[y, x]
            mag = np.hypot(dx, dy)
            if mag < min_magnitude:
                continue

            x2 = int(round(x + dx * scale))
            y2 = int(round(y + dy * scale))

            cv2.arrowedLine(
                image_bgr,
                (int(x), int(y)),
                (x2, y2),
                color=color,
                thickness=thickness,
                tipLength=tip_length,
            )

    return image_bgr


def create_optical_flow_video(
    frames_gray,
    flows,
    output_path="optical_flow_overlay.mp4",
    fps=25,
    alpha=0.45,
    arrow_step=24,
    arrow_scale=2.0,
    min_arrow_magnitude=0.5,
    codec="mp4v",
    fix_heatmap_scale=True,
):
    """
    Create a video:
      grayscale background + heatmap overlay + flow arrows

    Parameters
    ----------
    frames_gray : list or np.ndarray
        Sequence of grayscale frames, each shape (H, W), dtype uint8 preferred.
        Number of frames should be len(flows) + 1 or len(flows).
        If len(frames_gray) == len(flows) + 1, flow[i] is drawn on frame[i].
    flows : list or np.ndarray
        Sequence of optical flow arrays, each shape (H, W, 2).
    """
    if len(frames_gray) == len(flows) + 1:
        bg_frames = frames_gray[:-1]
    elif len(frames_gray) == len(flows):
        bg_frames = frames_gray
    else:
        raise ValueError("len(frames_gray) must be len(flows) or len(flows)+1")

    h, w = bg_frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for {output_path}")

    # Optional global heatmap scaling for temporal consistency
    global_vmax = None
    if fix_heatmap_scale:
        mags = [np.linalg.norm(flow, axis=2) for flow in flows]
        global_vmax = np.percentile(np.concatenate([m.ravel() for m in mags]), 99) + 1e-6

    for gray, flow in zip(bg_frames, flows):
        if gray.dtype != np.uint8:
            gray_u8 = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            gray_u8 = gray

        # grayscale -> BGR
        base = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)

        # magnitude heatmap
        mag = np.linalg.norm(flow, axis=2)
        heatmap = flow_to_heatmap(mag, vmax=global_vmax)

        # alpha blend
        overlay = cv2.addWeighted(base, 1.0 - alpha, heatmap, alpha, 0.0)

        # arrows
        overlay = draw_flow_arrows(
            overlay,
            flow,
            step=arrow_step,
            scale=arrow_scale,
            min_magnitude=min_arrow_magnitude,
            color=(255, 255, 255),
            thickness=1,
            tip_length=0.25,
        )

        writer.write(overlay)

    writer.release()
    print(f"Saved: {output_path}")
```

```python
def compute_farneback_flows(frames_gray):
    flows = []
    for i in tqdm(range(len(frames_gray) - 1), total=len(frames_gray) - 1, desc="Computing optical flows"):
        prev = frames_gray[i]
        nxt = frames_gray[i + 1]

        flow = cv2.calcOpticalFlowFarneback(
            prev, nxt, None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        flows.append(flow)
    return flows

flows = compute_farneback_flows(frames)
```

```python
create_optical_flow_video(
    frames_gray=frames,
    flows=flows,
    output_path="/Users/gimli/cvr/data/zavity/ETE 2026_03_16-all/out/GX011155-farneback.mp4",
    fps=30,
    alpha=0.40,
    arrow_step=28,
    arrow_scale=2.5,
    min_arrow_magnitude=0.7,
    codec="mp4v",
    fix_heatmap_scale=True,
)
```
