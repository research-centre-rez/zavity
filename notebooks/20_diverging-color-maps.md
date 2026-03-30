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
ROOT = "/Users/gimli/cvr/data/zavity/ETE 2026_03_16-all"
```

```python
import os
import imageio.v3 as iio
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tqdm.auto import tqdm
```

```python
oios_path = []
for root, dirs, files in os.walk(ROOT):
    for file in files:
        if file.endswith("oio-full.png"):
            oios_path.append(os.path.join(root, file))
```

```python
oios_path
```

```python
oio = iio.imread(oios_path[0])
plt.imshow(oio, cmap="gray")
plt.show()
```

## Estimating Rotation

```python
rots = []
erange = np.arange(-1.9, -1.7, 0.01) # Range should be set according to the actual data
for angle in erange:
    mat = cv2.getRotationMatrix2D((oio.shape[1] / 2, oio.shape[0] / 2), angle, 1)
    rotated = cv2.warpAffine(oio, mat, oio.shape[::-1])
    rots.append(rotated)
```

```python
variance = [np.sum(np.std(rot[500:6500], axis=1)) for rot in rots]
plt.plot(erange, variance)
plt.axvline(erange[np.argmin(variance)], color="red", label=f"minimum {erange[np.argmin(variance)]:.2f}")
plt.legend()
plt.show()
```

```python
mat = cv2.getRotationMatrix2D((oio.shape[1] / 2, oio.shape[0] / 2), -1.79, 1)
rotated = cv2.warpAffine(oio, mat, oio.shape[::-1])
plt.figure(figsize=(15, 10))
plt.imshow(np.dot(np.median(rotated, axis=1).reshape(-1, 1), np.ones((1, rotated.shape[1]))) , cmap="gray")
plt.show()
```

```python
window = 301
anomaly = np.zeros_like(rotated, dtype=float)
mean = np.zeros_like(rotated, dtype=float)
for column in tqdm(range(rotated.shape[1])):
    mean[:, column] = np.mean(rotated[:, np.max([0, column-window]): np.min([column + window, oio.shape[1]])], axis=1)
    anomaly[:, column] = rotated[:, column] - mean[:, column]
```

```python
plt.figure(figsize=(15, 10))
plt.imshow(mean, cmap="gray")
plt.show()
```

```python
plt.figure(figsize=(15, 10))
plt.imshow(anomaly>20, cmap="gray")
plt.show()
```

```python
from skimage.morphology import label
```

```python
import numpy as np
from scipy import ndimage

def remove_small_segments(binary_img, min_size):
    """
    binary_img: boolean or {0,1} array
    min_size: minimum number of pixels to keep
    """

    labeled, num = ndimage.label(binary_img)

    sizes = ndimage.sum(binary_img, labeled, range(1, num + 1))

    mask_sizes = sizes >= min_size
    mask_sizes = np.insert(mask_sizes, 0, False)  # background

    output = mask_sizes[labeled]
    return output.astype(binary_img.dtype)
```

```python
plt.figure(figsize=(15, 10))
opened_above = cv2.morphologyEx((anomaly>20).astype(np.uint8), cv2.MORPH_OPEN, (21,21))
opened_below = cv2.morphologyEx((anomaly<-20).astype(np.uint8), cv2.MORPH_OPEN, (21,21))
oa = remove_small_segments(opened, 580)
ob = remove_small_segments(opened_below, 580)
plt.imshow(np.logical_or(oa,ob) * anomaly, cmap="gray")
plt.show()
```

```python
plt.hist(anomaly.reshape(-1))
plt.show()
```

```python
sat = (np.logical_and(oa, ob).astype(np.float32) * 2 - 1) * anomaly.astype(np.float32) / np.abs(anomaly).max()
h2 = sat < 0
hue = np.ones((rotated.shape[0], rotated.shape[1]), np.float32) * 15
hue[h2] = 0
hsv = np.stack([
    hue * np.pi * 2,
    np.clip(np.abs(sat) * 2, 0, 1),
    (rotated[:, :rotated.shape[1]].astype(np.float32) / 255 / 2 + 0.5)
], axis=2)
```

```python
plt.figure(figsize=(15, 10))
plt.imshow(cv2.cvtColor(hsv.astype(np.float32), cv2.COLOR_HSV2RGB))
plt.show()
```

## Batch

```python
SEGMENT_AREA_MIN_THRESHOLD_PX = 580
INTENSITY_THRESHOLD = 20
SLIDING_WINDOW_SIZE = 301

for path in oios_path:
    print(f"Coloring oio at {path}")
    oio = iio.imread(path)
    rots = []
    erange = np.arange(-1.9, -1.7, 0.01) # Range should be set according to the actual data
    for angle in erange:
        mat = cv2.getRotationMatrix2D((oio.shape[1] / 2, oio.shape[0] / 2), angle, 1)
        rotated = cv2.warpAffine(oio, mat, oio.shape[::-1])
        rots.append(rotated)
    variance = [np.sum(np.std(rot[500:6500], axis=1)) for rot in rots]
    angle = erange[np.argmin(variance)]
    print(f"Estimated rotation angle: {angle:.2f}")
    mat = cv2.getRotationMatrix2D((oio.shape[1] / 2, oio.shape[0] / 2), angle, 1)
    rotated = cv2.warpAffine(oio, mat, oio.shape[::-1])

    anomaly = np.zeros_like(rotated, dtype=float)
    mean = np.zeros_like(rotated, dtype=float)
    for column in tqdm(range(rotated.shape[1])):
        mean[:, column] = np.mean(rotated[:, np.max([0, column-SLIDING_WINDOW_SIZE]): np.min([column + SLIDING_WINDOW_SIZE, oio.shape[1]])], axis=1)
        anomaly[:, column] = rotated[:, column] - mean[:, column]

    opened_above = cv2.morphologyEx((anomaly> INTENSITY_THRESHOLD).astype(np.uint8), cv2.MORPH_OPEN, (21,21))
    opened_below = cv2.morphologyEx((anomaly<-INTENSITY_THRESHOLD).astype(np.uint8), cv2.MORPH_OPEN, (21,21))
    oa = remove_small_segments(opened_above, SEGMENT_AREA_MIN_THRESHOLD_PX)
    ob = remove_small_segments(opened_below, SEGMENT_AREA_MIN_THRESHOLD_PX)

    sat = (np.logical_and(oa, ob).astype(np.float32) * 2 - 1) * anomaly.astype(np.float32) / np.abs(anomaly).max()
    h2 = sat < 0
    hue = np.ones((rotated.shape[0], rotated.shape[1]), np.float32) * 15
    hue[h2] = 0
    hsv = np.stack([
        hue * np.pi * 2,
        np.clip(np.abs(sat) * 2, 0, 1),
        (rotated[:, :rotated.shape[1]].astype(np.float32) / 255 / 2 + 0.5)
    ], axis=2)
    diverging_colormap = cv2.cvtColor(hsv.astype(np.float32), cv2.COLOR_HSV2RGB)
    print("Writing the image to disk...")
    iio.imwrite(path.replace("oio-full", "diverging-colormap"), (diverging_colormap * 255).astype(np.uint8))
```

```python
(np.logical_and(oa, ob).astype(np.float32) * 2 - 1) * anomaly.astype(np.float32) / np.abs(anomaly).max()
```

```python

```
