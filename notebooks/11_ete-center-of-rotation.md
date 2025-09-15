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
import scipy.signal
```

```python
vidcap = cv2.VideoCapture("/Users/gimli/cvr/data/zavity/ETE 2025_07_01-sample/in/Hor_ZH2_down.MP4")
```

```python
_, frame = vidcap.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
```

```python
import matplotlib.pyplot as plt
```

```python
rotate_matrix = cv2.getRotationMatrix2D((frame.shape[1] / 2 + 270, frame.shape[0]/2 + 50), 90, 1)
```

```python
rotated_image = cv2.warpAffine(
                src=frame,
                M=rotate_matrix,
                dsize=(frame.shape[1], frame.shape[0]),
                flags=cv2.INTER_CUBIC
            )
```

```python
from skimage.util import compare_images
```

```python
plt.imshow(compare_images(frame, rotated_image, method="blend"), cmap="gray")
plt.show()
```

```python
plt.figure(figsize=(15, 10))
ax = plt.subplot(1,2,1)
ax.imshow(frame)
ax.axhline(frame.shape[0]/2 + 50, color="red")
ax.axhline(frame.shape[0]/2 + 50 - 800, color="black")
ax.axhline(frame.shape[0]/2 + 50 + 800, color="black")
ax.axhline(frame.shape[0]/2 + 50 - 1300, color="red")
ax.axhline(frame.shape[0]/2 + 50 + 1300, color="red")
ax.axvline(frame.shape[1]/2 + 270, color="red")
ax.axvline(frame.shape[1]/2 + 270 - 800, color="black")
ax.axvline(frame.shape[1]/2 + 270 + 800, color="black")
ax.axvline(frame.shape[1]/2 + 270 - 1300, color="red")
ax.axvline(frame.shape[1]/2 + 270 + 1300, color="red")
ax = plt.subplot(1,2,2)
ax.imshow(rotated_image)
plt.show()
```

```python
features = cv2.goodFeaturesToTrack(frame, 500, 0.01, 50)
```

```python
import numpy as np
from scipy.optimize import minimize
```

```python
def circle_error(center):
    return np.sum(
        np.sqrt(np.power(features[:, 0, 1] - center[1], 2) + np.power(features[:, 0, 0] - center[0], 2)) > radius
    )

```

```python
radius = 1200
circle_position = minimize(circle_error, x0=np.array(frame.shape)[::-1] / 2, method="Nelder-Mead")
```

```python
ax = plt.subplot(111)
ax.imshow(frame, cmap="gray")
ax.scatter(features[:, 0 , 0], features[:, 0, 1], marker="+", color="red")
circle = plt.Circle(circle_position.x, radius, alpha=0.2, color="purple")
ax.add_patch(circle)
plt.show()
```

```python
from tqdm.auto import tqdm
```

```python
success = True
centers = [np.array(frame.shape)[::-1] / 2]
writer = cv2.VideoWriter("/Users/gimli/cvr/data/zavity/ETE 2025_07_01-sample/out/centers.mp4",
    apiPreference=cv2.CAP_FFMPEG,
    fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
    fps=vidcap.get(cv2.CAP_PROP_FPS),
    frameSize=(1300, 1300),
    params=[
        cv2.VIDEOWRITER_PROP_DEPTH,
        cv2.CV_8U,
        cv2.VIDEOWRITER_PROP_IS_COLOR,
        0,
    ])
vidcap = cv2.VideoCapture("/Users/gimli/cvr/data/zavity/ETE 2025_07_01-sample/in/Hor_ZH2_down.MP4")
for frame_id in tqdm(range(int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)))):
    success, frame = vidcap.read()
    if not success:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    features = cv2.goodFeaturesToTrack(frame, 500, 0.01, 50)
    circle_position = minimize(circle_error, x0=centers[-1], method="Nelder-Mead")
    centers.append(circle_position.x)
    writer.write(frame[int(centers[-1][1]) - 650:int(centers[-1][1]) + 650, int(centers[-1][0]) - 650: int(centers[-1][0]) + 650])
writer.release()
```

```python
plt.figure(figsize=(15, 15))
plt.imshow(frame, cmap="gray")
plt.scatter(np.array(centers)[:,0], np.array(centers)[:,1], color="red", alpha=0.01)
plt.show()
```

```python
from scipy.signal import savgol_filter
```

```python
plt.figure(figsize=(15,5))
plt.plot(np.array(centers)[:,0])
plt.plot(savgol_filter(np.array(centers)[:,0], 100, 1))
plt.show()
```

```python
plt.plot(np.array(centers)[:,1])
plt.show()
```

```python
plt.figure(figsize=(15,5))
ax = plt.subplot(111)
ax.plot(savgol_filter(np.angle((centers - np.mean(centers, axis=0))[:,0].astype(complex) + (centers - np.mean(centers, axis=0))[:,1].astype(complex) * 1j), 10, 1))
ax = ax.twinx()
ax.plot(savgol_filter(np.abs((centers - np.mean(centers, axis=0))[:,0].astype(complex) + (centers - np.mean(centers, axis=0))[:,1].astype(complex) * 1j), 100, 1), color="red")
plt.show()
```

```python
np.angle(np.array([-1,-1]).T)
```

```python
plt.plot(np.abs(centers - np.median(centers)))
plt.show()
```

```python
centers_shaky = np.copy(centers)
```

```python
centers_stable = np.stack([savgol_filter(np.array(centers)[:,0], 100, 1), savgol_filter(np.array(centers)[:,1], 100, 1)], axis=1)
```

```python
writer = cv2.VideoWriter("/Users/gimli/cvr/data/zavity/ETE 2025_07_01-sample/out/stable.mp4",
    apiPreference=cv2.CAP_FFMPEG,
    fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
    fps=vidcap.get(cv2.CAP_PROP_FPS),
    frameSize=(1300, 1300),
    params=[
        cv2.VIDEOWRITER_PROP_DEPTH,
        cv2.CV_8U,
        cv2.VIDEOWRITER_PROP_IS_COLOR,
        0,
    ])
vidcap = cv2.VideoCapture("/Users/gimli/cvr/data/zavity/ETE 2025_07_01-sample/in/Hor_ZH2_down.MP4")
for frame_id in tqdm(range(int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)))):
    success, frame = vidcap.read()
    if not success:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    writer.write(frame[int(centers_stable[frame_id][1]) - 650:int(centers_stable[frame_id][1]) + 650, int(centers_stable[frame_id][0]) - 650: int(centers_stable[frame_id][0]) + 650])
writer.release()
```

```python
centers_stable
```

```python
from matplotlib.collections import LineCollection

points = np.array([centers[:, 0], centers[:, 1]]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
norm = plt.Normalize(0, len(centers_stable) - 1)
colors = plt.cm.viridis(norm(np.arange(len(segments))))
lc = LineCollection(segments, colors=colors, linewidth=2, alpha=0.8)
lc.set_array(np.linspace(0, 1, len(centers_stable)))
fig, ax = plt.subplots()
ax.add_collection(lc)
ax.autoscale()
ax.set_aspect("equal")
plt.show()
```

```python
%load_ext autoreload
%autoreload 2
```

```python
from src.steps.adaptive_frame_cropping import AdaptiveFrameCropper
```

```python
afc = AdaptiveFrameCropper("/Users/gimli/cvr/data/zavity/ETE 2025_07_01-sample/in/Hor_ZH2_down.MP4")
```

```python
centers = afc.get_frames_center()
```

```python
import pandas as pd
import matplotlib.pyplot as plt
```

```python
centers = pd.read_csv("/Users/gimli/cvr/data/zavity/ETE 2025_07_01-sample/out/Hor_ZH2_down-frameCenters.csv").to_numpy()
angles = pd.read_csv("/Users/gimli/cvr/data/zavity/ETE 2025_07_01-sample/out/Hor_ZH2_down-full_angles.csv").to_numpy()
```

```python
from matplotlib.collections import LineCollection

points = np.array([centers[:, 1], centers[:, 2]]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
norm = plt.Normalize(0, len(centers) - 1)
colors = plt.cm.viridis(norm(np.arange(len(segments))))
lc = LineCollection(segments, colors=colors, linewidth=2, alpha=0.8)
lc.set_array(np.linspace(0, 1, len(centers)))
fig, ax = plt.subplots()
ax.add_collection(lc)
ax.autoscale()
ax.set_aspect("equal")
plt.show()
```

```python
centers_distance = np.sqrt(np.power(np.diff(centers[:, 1]),2) + np.power(np.diff(centers[:, 2]),2))
```

```python
from scipy.signal import savgol_filter
```

```python
plt.figure(figsize=(15,5))
plt.plot(savgol_filter(angles[0:1000, 1], 5, 1))
plt.show()
```

```python
import cv2
from steps.adaptive_frame_cropping import AdaptiveFrameCropper, CROPPED_FRAME_SIDE_PX
```

```python
vidcap = cv2.VideoCapture("/Users/gimli/cvr/data/zavity/ETE 2025_07_01-sample/in/Hor_ZH2_down.MP4")
writer = cv2.VideoWriter("/Users/gimli/cvr/data/zavity/ETE 2025_07_01-sample/out/stable-rot.mp4",
    apiPreference=cv2.CAP_FFMPEG,
    fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
    fps=vidcap.get(cv2.CAP_PROP_FPS),
    frameSize=(CROPPED_FRAME_SIDE_PX, CROPPED_FRAME_SIDE_PX),
    params=[
        cv2.VIDEOWRITER_PROP_DEPTH,
        cv2.CV_8U,
        cv2.VIDEOWRITER_PROP_IS_COLOR,
        0,
    ])
a = savgol_filter(angles[:, 1], 5, 1)
frame_no = 0
for i in range(1000):
    success, frame = vidcap.read()
    if not success:
        break
    cx, cy = centers[frame_no, 1:]
    angle = a[frame_no]
    rotation_matrix = cv2.getRotationMatrix2D((int(CROPPED_FRAME_SIDE_PX // 2), int(CROPPED_FRAME_SIDE_PX // 2)), angle, 1.0)
    cropped_frame = cv2.warpAffine(AdaptiveFrameCropper.crop(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cx, cy), rotation_matrix, (CROPPED_FRAME_SIDE_PX, CROPPED_FRAME_SIDE_PX))
    #cropped_frame = AdaptiveFrameCropper.crop(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cx, cy)
    writer.write(cropped_frame)
    frame_no += 1

vidcap.release()
writer.release()
```

```python


```
