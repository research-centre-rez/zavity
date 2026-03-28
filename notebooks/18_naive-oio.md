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

# Stitching of preprocessed video

- row stitching
- row y-compensation (necessary due to the tubus and socket axes mismatch)
- OIO build from rows

This notebook was used for dataset ETE 2026/03/16.

```python
%load_ext autoreload
%autoreload 2
```

```python
import os
import numpy as np
import pandas as pd
import cv2
from tqdm.auto import tqdm
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
from src.steps.image_row_builder import ImageRowBuilder
```

```python
configurations = [
    ("/Users/gimli/cvr/data/zavity/ETE 2026_03_16-all/GX011155", False),
    ("/Users/gimli/cvr/data/zavity/ETE 2026_03_16-all/GX011157", False),
    ("/Users/gimli/cvr/data/zavity/ETE 2026_03_16-all/GX011159", True),
    ("/Users/gimli/cvr/data/zavity/ETE 2026_03_16-all/GX011161", True),
    ("/Users/gimli/cvr/data/zavity/ETE 2026_03_16-all/GX011163", False),
    ("/Users/gimli/cvr/data/zavity/ETE 2026_03_16-all/GX011165", False),
    ("/Users/gimli/cvr/data/zavity/ETE 2026_03_16-all/GX011167", False),
    ("/Users/gimli/cvr/data/zavity/ETE 2026_03_16-all/GX011169", True),
    ("/Users/gimli/cvr/data/zavity/ETE 2026_03_16-all/GX011171", True),
    ("/Users/gimli/cvr/data/zavity/ETE 2026_03_16-all/GX011173", False)
]
```

```python
def load_and_sort_videos(ROOT):
    video_filename = os.path.basename(ROOT)
    intervals =  pd.read_csv(os.path.join(ROOT, f"{video_filename}-breakpoints.csv"))
    intervals = intervals[intervals["sequence type (rot 1, shift 0)"]==1].to_numpy().astype(int)

    rot_videos = []
    for file in os.listdir(ROOT):
        for interval in intervals:
            if f"{interval[0]:05d}" in file:
                print(f"{file} => {interval[0]}")
                rot_videos.append(os.path.join(ROOT, file))

    return video_filename, sorted(rot_videos)
```

```python
def load_frames(scan_cap, rotate=False):
    end_frame = int(scan_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scans = []
    for _ in tqdm(range(end_frame), desc="Loading frames"):
        success, frame = scan_cap.read()
        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            # warped = warp_image_with_tps(im2, imgwarp, im2.shape)
            # no other preprocessing
            scans.append(frame)

    scan_cap.release()
    return scans
```

```python
def stitch_frames(images, x_shifts, y_shifts, frame_slice_start, averaged_pixels):
    height, width = images[0].shape

    x_shifts_cum = np.cumsum(x_shifts)
    x_zero = np.min(x_shifts_cum)
    x_shifts_cum -= x_zero
    oio_height = int(np.sum(y_shifts))
    oio_width = int(np.max(x_shifts_cum) - np.min(x_shifts_cum) + width)

    accumulator = np.zeros((np.abs(oio_height), oio_width)).astype(float)
    weights = np.zeros((np.abs(oio_height), oio_width))
    position = 0 if oio_height > 0 else np.abs(oio_height) + y_shifts[0]
    for idx, image in tqdm(enumerate(images), total=len(images), desc="Stitching frames"):
        slice_start = np.floor(image.shape[0] * frame_slice_start).astype(int) - 1
        slice_height = np.abs(np.sum(y_shifts[idx: np.min([idx + averaged_pixels, y_shifts.size])]).astype(int))
        if position + slice_height > np.abs(oio_height):
            slice_height = np.abs(oio_height) - position
        slice_end = np.ceil(slice_start + slice_height).astype(int) + 1

        row_px_position = np.arange(-(x_shifts_cum[idx] % 1), -(x_shifts_cum[idx] % 1) + image.shape[1], 1)
        acc_row_px_position = slice(int(x_shifts_cum[idx]), int(x_shifts_cum[idx]) + image.shape[1])

        acc_target = np.arange(np.floor(position).astype(int), np.ceil(position + slice_height).astype(int))
        extra_line = int(len(acc_target) - (slice_end - slice_start - 2))  # there is extra line
        row_weight = np.array(
            [1 - (position % 1)] + [1 for _ in range(acc_target.size - 2)] +
            [(position + slice_height) % 1]
        ).reshape(-1, 1)[:accumulator[acc_target, acc_row_px_position].shape[0]]
        #for rgb in range(3):
        interpolated = RectBivariateSpline(x=np.arange(slice_start, slice_end),
                                           y=np.arange(image.shape[0]),
                                           z=image[slice_start: slice_end, :],
                                           kx=np.min([slice_end - slice_start - 1, 3]))
        accumulator[acc_target, acc_row_px_position] += interpolated(
            np.arange(slice_start + 1, slice_end - 1 + extra_line), row_px_position) * row_weight
        weights[acc_target, acc_row_px_position] += np.ones((acc_target.size, row_px_position.size)) * row_weight

        position += y_shifts[idx]

    oio = accumulator
    # for rgb in range(3):
    oio[weights != 0] = accumulator[weights != 0] / weights[weights != 0]
    oio = ((oio - np.min(oio)) * 255 / (np.max(oio) - np.min(oio))).astype(np.uint8)
    return oio
```

```python
def build_raw_oio(video, rotate=False):
    scans = load_frames(cv2.VideoCapture(video), rotate=rotate)
    oio_raw = stitch_frames([scan.T for scan in scans],
             x_shifts=np.zeros((len(scans))),
             y_shifts= - 15 * np.ones((len(scans))),
             frame_slice_start=0.4,
             averaged_pixels=2).T
    return oio_raw
```

```python
def y_compensation(oio_raw):
    margin = oio_raw.shape[1] - width
    img = oio_raw[:, margin // 2 : -(margin - (margin // 2))]
    y_comp = ImageRowBuilder.compute_column_shifts_dic(img)
    return ImageRowBuilder.remove_column_shifts(img, y_comp)
```

```python
for ROOT, rotate in configurations:
    video_filename, rot_videos = load_and_sort_videos(ROOT)
    oios_raw = []
    for vid, video in enumerate(rot_videos):
        oios_raw.append(build_raw_oio(video, rotate=rotate))

    print("Counting crop: ")
    roll = 928 - 1678
    blend = 1370 - 1860
    crop = 1585 - 910
    width = np.min([oio.shape[1] for oio in oios_raw]) - crop

    oios = []
    for vid, oio_raw in enumerate(oios_raw):
        oio = y_compensation(oio_raw)
        oios.append(oio)
        cv2.imwrite(os.path.join(ROOT, f"{video_filename}-oio-{vid:02d}.png"), np.roll(oio, 500, axis=1))

    height_minus = (len(oios) - 1) * blend
    full_oio = np.zeros((width, np.sum([oio.shape[0] for oio in oios]) + height_minus)).T
    weight = np.zeros((width, np.sum([oio.shape[0] for oio in oios]) + height_minus)).T
    print(f"Width: {width} - {full_oio.shape}")

    print("Full OIO build:")
    row = 0
    for oid, oio in enumerate(oios):
        print(oio.shape, row)
        full_oio[row: row + oio.shape[0], :] += np.roll(oio, oid * roll)
        weight[row: row + oio.shape[0], :] += np.ones((oio.shape[0], width))
        row += oio.shape[0] + blend
    full_oio /= weight

    cv2.imwrite(os.path.join(ROOT, f"{video_filename}-oio-full.png"), full_oio.astype(np.uint8))

```

```python
img = -1
plt.figure(figsize=(6, 15))
ax = plt.subplot(1, 4, 1)
ax.imshow(np.roll(oios_raw[img][:, -250:], -30, axis=0), cmap="gray")
ax = plt.subplot(1, 4, 2)
ax.imshow(oios_raw[img][:, :250], cmap="gray")

ax = plt.subplot(1, 4, 3)
ax.imshow(np.roll(oios[img][:, -250:], 0, axis=0), cmap="gray")
ax = plt.subplot(1, 4, 4)
ax.imshow(oios[img][:, :250], cmap="gray")
plt.show()
```

```python
row = 0
for oid, oio in enumerate(oios[-6:]):
    print(oio.shape, row)
    full_oio[row: row + oio.shape[0], :] += np.roll(oio, oid * roll)
    weight[row: row + oio.shape[0], :] += np.ones((oio.shape[0], width))
    row += oio.shape[0] + blend
full_oio /= weight
```

```python
plt.figure(figsize=(15, 15))
plt.imshow(full_oio, cmap="gray")
plt.show()
```

```python
cv2.imwrite(os.path.join(ROOT, f"{video_filename}-oio-full.png"), full_oio.astype(np.uint8))
```

```python

```
