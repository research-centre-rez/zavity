---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import cv2
import imageio.v3 as iio
import numpy as np
from scipy.optimize import minimize, brute
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.ndimage import median_filter
import cv2
```

```python
img = iio.imread("/Users/gimli/cvr/data/zavity/ETE 2025_07_01-sample/out/Hor_ZH2_down-oio-0.png")
blurred = cv2.GaussianBlur(img,(101,51),121)
```

```python
plt.figure(figsize=(15,8))
plt.imshow(cv2.resize(blurred, (3000, blurred.shape[0])), cmap="gray")
plt.show()
```

```python
# This is continuous version of image columns registration - i.e. shift in y can be float
I = blurred
yshifts = [0]
for column in tqdm(range(1, I.shape[1])):
    def yshift(y):
        sum = 0
        spline = CubicSpline(x=np.arange(I.shape[0]), y=I[:, column])
        for en, yshifts_len in enumerate([100, 10]):
            if len(yshifts) > yshifts_len:
                relative_shift = np.sum(yshifts[-yshifts_len:]) + y[0]
                sampled = spline(np.arange(600 + relative_shift, 1200 + relative_shift, 1)[:600])
                sum -= (en + 1) * np.sum(sampled * I[600: 1200, column - yshifts_len])

        relative_shift = y[0]
        sampled = spline(np.arange(600 + relative_shift, 1200 + relative_shift, 1)[:600])
        sum -= 3 * np.sum(sampled * I[600: 1200, column - 1])

        return sum

    # Here is the problem (most probable is the interpolation) because it returns more often positive values.
    # Cummulative sum of shifts then shifts whole image upwards without clear reason
    dic = minimize(yshift, x0=[0], method="Nelder-Mead", bounds=[(-1, 1)])
    if dic.success:
        yshifts.append(dic.x[0])
    else:
        yshifts.append(0)
```

```python
I = blurred
yshifts = [0]
yshifts_cum = [0]
for column in tqdm(range(1, I.shape[1])):
    def yshift(y):
        sum = 0
        for en, yshifts_len in enumerate(np.arange(10, 1000, 100)):
            if len(yshifts) > yshifts_len:
                relative_shift = int(yshifts_cum[-1] - yshifts_cum[-yshifts_len] + y[0])
                sampled = I[600 + relative_shift: 1200 + relative_shift, column]
                sum += np.sum(np.abs(np.diff(sampled) - np.diff(I[600: 1200, column - yshifts_len])))
            else:
                break

        # relative_shift = int(y[0])
        # sampled = I[600 + relative_shift: 1200 + relative_shift, column]
        # sum -= np.sum(sampled * I[600: 1200, column - 1])

        return sum

    dic = brute(yshift, ranges=[slice(-1,2,1)])
    yshifts.append(dic[0])
    yshifts_cum.append(yshifts_cum[-1] + dic[0])

```

```python
# Above code works well for columns 1000+. To fix the left part of the image we run the same code backwards for these columns
for column in tqdm(np.arange(1000, -1, -1)):
    def yshift(y):
        sum = 0
        for en, yshifts_len in enumerate(np.arange(column + 10, column + 1000, 100)):
            relative_shift = int(yshifts_cum[column + 1] - yshifts_cum[yshifts_len] + y[0])
            sampled = I[600 + relative_shift: 1200 + relative_shift, column]
            sum += np.sum(np.abs(np.diff(sampled) - np.diff(I[600: 1200, yshifts_len])))

        return sum

    dic = brute(yshift, ranges=[slice(-1,2,1)])
    yshifts[column] = -dic[0]
    yshifts_cum[column] = yshifts_cum[column + 1] + dic[0]
```

```python
np.polyfit(np.arange(len(yshifts_smooth)), yshifts_smooth, deg=1)
```

```python
plt.plot(np.cumsum(yshifts))
yshifts_smooth = median_filter(np.cumsum(yshifts), 1001, mode="nearest")  # It is not clear what is a good smoothing factor. Large numbers have a problem with the beginning and the end of a signal
plt.plot(yshifts_smooth)
plt.plot(np.arange(len(yshifts_smooth)), np.polyval(np.polyfit(np.arange(len(yshifts_smooth)), yshifts_smooth, deg=1), np.arange(len(yshifts_smooth))), color="red")
plt.show()
```

```python
row = np.zeros_like(img)
for column in np.arange(img.shape[1]):
    row[:, column] = np.roll(img[:, column], -yshifts_smooth[column].astype(int))
```

```python
crop = np.max(np.abs(-yshifts_smooth[column].astype(int)))
```

```python
plt.figure(figsize=(15, 8))
ax = plt.subplot(121)
ax.imshow(cv2.resize(img, (1400, row.shape[0])), cmap="gray")
ax = plt.subplot(122)
ax.imshow(cv2.resize(row, (1400, row.shape[0]))[crop: -crop], cmap="gray")
plt.show()
```

```python
plt.figure(figsize=(15, 8))
plt.imshow(row[crop: -crop], cmap="gray")
plt.xlim(11000)
plt.show()
```

```python
pos = np.load("/Users/gimli/cvr/data/zavity/ETE 2025_07_01-sample/out/Hor_ZH2_down-positions.npy")
```

```python
pos
```

```python
speeds = np.load("/Users/gimli/cvr/data/zavity/ETE 2025_07_01-sample/out/Hor_ZH2_down-preprocessed-speeds.npy", allow_pickle=True)
```

```python
speeds
```

```python

```
