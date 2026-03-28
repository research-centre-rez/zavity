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
import imageio.v3 as iio
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import matplotlib.pyplot as plt
import tempfile
import subprocess
import cv2
import numpy as np
from tqdm.auto import tqdm
```

```python
oio = iio.imread("/Users/gimli/cvr/data/zavity/GoPro-skeny/Cely_sken_3_110proc/oio.png")
```

```python
plt.imshow(oio)
plt.show()
```

```python
otsu_threshold, _ = cv2.threshold(oio, 0, 255, cv2.THRESH_OTSU)
```

```python
with tempfile.NamedTemporaryFile(suffix=".pgm", delete=False) as tmpfile:
    filename = tmpfile.name
    iio.imwrite(tmpfile.name, oio)  # this must be a grayscale image
    process = subprocess.Popen(
        ["/Users/gimli/projects/scripts/devernay", tmpfile.name,
         "-t", "/dev/stdout",
         "-l", f"{otsu_threshold / 15}",
         "-h", f"{otsu_threshold / 3}",
         "-p", "/Users/gimli/sample.pdf",
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

filtered = np.rad2deg(np.abs(angles[~np.isnan(angles)]))
```

```python
count, values = np.histogram(filtered, bins=900)
```

```python
rot = cv2.getRotationMatrix2D((oio.shape[1]/2, oio.shape[0]/2), values[np.argmax(count)], 1)
```

```python
oio_rotated = cv2.warpAffine(oio, rot, (oio.shape[1], oio.shape[0]))
```

```python
plt.figure(figsize=(15, 10))
plt.imshow(oio_rotated, cmap="gray")
plt.axhline(oio.shape[0]//2, color="red")
plt.xticks([])
plt.show()
```

```python
window = 701
anomaly = np.zeros_like(oio_rotated, dtype=float)
for column in tqdm(range(8000)):
    anomaly[:, column] = oio_rotated[:, column] - np.mean(oio_rotated[:, np.max([0, column-window]): np.min([column + window, oio.shape[1]])], axis=1)
```

```python
plt.figure(figsize=(15, 10))
plt.imshow(anomaly[:,:8000], cmap="gray")
plt.show()
```

```python
n = ((anomaly[:, :8000] - np.min(anomaly[:, :8000])) / (np.max(anomaly[:, :8000]) - np.min(anomaly[:, :8000])) * 255 * 255).astype(np.uint16)
clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(30,30))
img2 = clahe.apply(n)
```

```python
sat = (img2 / 255.0 / 255.0).astype(np.float32) - 0.5
h2 = sat < 0
hue = np.ones((oio_rotated.shape[0], 8000), np.float32) * 15
hue[h2] = 0
```

```python
hsv = np.stack([
    hue * np.pi * 2, 
    np.clip(np.abs(sat) * 2, 0, 1),
    (oio_rotated[:, :8000].astype(np.float32) / 255 / 2 + 0.5)
], axis=2)
```

```python
plt.imshow(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
plt.show()
```

```python
iio.imwrite("/Users/gimli/anomaly.png", cv2.resize(cv2.cvtColor(hsv[:8000], cv2.COLOR_HSV2RGB) * 255, (1200, 1200)).astype(np.uint8))
```

```python
iio.imwrite("/Users/gimli/raw.png", cv2.resize(oio_rotated[:8000, :8000], (1200, 1200)).astype(np.uint8))
```

```python
angles = np.load("/Users/gimli/cvr/data/zavity/zavity-output/_threaded_socket-rectified-full_angles.npy")
```

```python
plt.plot(angles)
plt.show()
```

```python

```
