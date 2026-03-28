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
ROOT = "/Users/gimli/cvr/data/zavity/EDU 2025_04_24/sken-splitted-outputs"
```

```python
import imageio.v3 as iio
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import matplotlib.pyplot as plt
import tempfile
import subprocess
import cv2
import numpy as np
from tqdm.auto import tqdm
import os
```

Load all images prepared in a folder:

```python
oios = {}
for file in os.listdir(ROOT):
    if file.endswith("oio.png"):
        _, hid, _, pid, _ = file.split("-")
        oios[(hid, pid)] = iio.imread(os.path.join(ROOT, file))
```

```python
plt.figure(figsize=(15, 10))
plt.title("14-03 preview")
plt.imshow(oios[('14', '03')])
plt.show()
```

# Preprocessing

- apply devernay algorithm for subpixel edge detection
- compute the angle
- rotate the image

```python
def devernay(oio):
    with tempfile.NamedTemporaryFile(suffix=".pgm", delete=False) as tmpfile:
        iio.imwrite(tmpfile.name, oio)  # this must be a grayscale image
        process = subprocess.Popen(
            ["/Users/gimli/projects/scripts/devernay", tmpfile.name,
             "-t", "/dev/stdout",
             "-l", f"{otsu_threshold / 15}",
             "-h", f"{otsu_threshold / 3}",
             "-s", f"1",
             ], stdout=subprocess.PIPE)
        tmpfile.close()
    result = process.stdout.read().decode("utf-8")
    lines = result.split("\n")
    dev = []
    for line in lines:
        if line != "":
            x, y = line.split(' ')
            dev.append((float(x), float(y)))
    dev = np.array(dev)
    if len(dev) == 0:
        print("Something wrong happen")
    samples = 1000
    choice = np.random.randint(0, len(dev), samples)
    xx0 = np.matmul(dev[choice, 0].reshape(-1, 1), np.ones((1, len(choice))))
    yy0 = np.matmul(dev[choice, 1].reshape(-1, 1), np.ones((1, len(choice))))
    xx1 = np.matmul(np.ones((len(choice), 1)), dev[choice, 0].reshape(1, -1))
    yy1 = np.matmul(np.ones((len(choice), 1)), dev[choice, 1].reshape(1, -1))
    valid = np.zeros_like(xx0, dtype=bool)
    valid[xx0 != xx1] = 1
    angles = np.zeros_like(xx0, np.float32)
    angles[xx0 == xx1] = np.pi / 2
    angles[valid] = np.arctan((yy0[valid] - yy1[valid]) / (xx0[valid] - xx1[valid])).reshape(-1)
    angles[np.eye(samples, dtype=bool)] = np.nan

    return np.rad2deg(np.abs(angles[~np.isnan(angles)]))
```

```python
DEBUG = True
for key, oio in oios.items():
    otsu_threshold, _ = cv2.threshold(oio.astype(np.uint8), 0, 255, cv2.THRESH_OTSU)
    filtered = devernay(oio)
    # For higher precision increase number of bins
    count, values = np.histogram(filtered, bins=900)
    # the most frequent angle is the one we use:
    rot = cv2.getRotationMatrix2D((oio.shape[1]/2, oio.shape[0]/2), -values[np.argmax(count)], 1)
    print("Computed angle: ", -values[np.argmax(count)], " degrees. Rotating image by this angle.")
    oio_rotated = cv2.warpAffine(oio, rot, (oio.shape[1], oio.shape[0]))

    # To control that image is correctly rotated it can be visualized
    if DEBUG:
        plt.figure(figsize=(15, 5))
        ax = plt.subplot(1,3,1)
        ax.imshow(oio_rotated, cmap="gray")
        ax.axhline(oio.shape[0]//2, color="red")
        ax.set_xticks([])

    # For visualization of smaller defects use smaller window
    window = 701
    anomaly = np.zeros_like(oio_rotated, dtype=float)
    for column in tqdm(range(oio_rotated.shape[1])):
        anomaly[:, column] = oio_rotated[:, column] - np.mean(oio_rotated[:, np.max([0, column-window]): np.min([column + window, oio.shape[1]])], axis=1)

    # Visualize diff image
    if DEBUG:
        ax = plt.subplot(1,3,2)
        ax.imshow(anomaly, cmap="gray")
    # rescaled anomaly
    iio.imwrite(os.path.join(ROOT, f"hnizdo-{key[0]}-part-{key[1]}-abs-diff.png"), (np.abs(anomaly)).astype(np.uint8))
    iio.imwrite(os.path.join(ROOT, f"hnizdo-{key[0]}-part-{key[1]}-normalized-diff.png"), ((anomaly - np.min(anomaly)) * 255 / (np.max(anomaly) - np.min(anomaly))).astype(np.uint8))

    # rescale image
    n = ((anomaly - np.min(anomaly)) / (np.max(anomaly) - np.min(anomaly)) * 255 * 255).astype(np.uint16)
    # apply adaptive histogram equalization (highlight defects)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(30,30))
    img2 = clahe.apply(n)

    # Make colored image
    sat = (img2 / 255.0 / 255.0).astype(np.float32) - 0.5
    h2 = sat < 0
    hue = np.ones((oio_rotated.shape[0], oio_rotated.shape[1]), np.float32) * 15
    hue[h2] = 0
    hsv = np.stack([
        hue * np.pi * 2,
        np.clip(np.abs(sat) * 2, 0, 1),
        (oio_rotated[:, :oio_rotated.shape[1]].astype(np.float32) / 255 / 2 + 0.5)
    ], axis=2)

    if DEBUG:
        ax = plt.subplot(1,3,3)
        ax.imshow(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
        plt.show()

    iio.imwrite(os.path.join(ROOT, f"hnizdo-{key[0]}-part-{key[1]}-colored.png"), (cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) * 255).astype(np.uint8))
```

```python

```
