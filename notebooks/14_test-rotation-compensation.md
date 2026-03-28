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
import pandas as pd
import cv2
import numpy as np
```

```python

```

```python
centers = pd.read_csv("/Users/gimli/cvr/data/zavity/ETE 2025_07_01-sample/out/Hor_ZH2_down-frameCenters.csv").to_numpy()
```

```python
angles = pd.read_csv("/Users/gimli/cvr/data/zavity/ETE 2025_07_01-sample/out/Hor_ZH2_down-full_angles.csv").to_numpy()
```

```python
plt.figure(figsize=(15,5))
plt.plot(angles[:,1])
plt.plot(angles[:,2])
plt.show()
```

```python
vidcap = cv2.VideoCapture("/Users/gimli/cvr/data/zavity/ETE 2025_07_01-sample/in/Hor_ZH2_down.MP4")
```



```python
sequence = (8203, 9106)
```

```python
from steps.adaptive_frame_cropping import CROPPED_FRAME_SIDE_PX, AdaptiveFrameCropper
from tqdm.auto import tqdm
```

```python
writer = cv2.VideoWriter(
            f"/Users/gimli/cvr/data/zavity/ETE 2025_07_01-sample/out/seq{sequence[0]:05d}.mp4",
            apiPreference=cv2.CAP_FFMPEG,
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
            fps=vidcap.get(cv2.CAP_PROP_FPS),
            frameSize=(CROPPED_FRAME_SIDE_PX, CROPPED_FRAME_SIDE_PX),
            params=[
                cv2.VIDEOWRITER_PROP_DEPTH,
                cv2.CV_8U,
                cv2.VIDEOWRITER_PROP_IS_COLOR,
                0,
            ])

vidcap.set(cv2.CAP_PROP_POS_FRAMES, sequence[0])
for fno in tqdm(np.arange(sequence[0], sequence[1])):
    _, frame = vidcap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    M = cv2.getRotationMatrix2D(centers[fno, 1:], angles[fno,1], 1)
    rotated_image = AdaptiveFrameCropper.crop(cv2.warpAffine(frame, M, frame.shape[:2][::-1]), centers[fno, 1], centers[fno, 2])
    writer.write(rotated_image)

writer.release()
```

```python
import matplotlib.pyplot as plt
```

```python
M = cv2.getRotationMatrix2D(centers[fno, 1:], angles[fno,1], 1)
rotated_image = cv2.warpAffine(frame, M, frame.shape[:2][::-1])
plt.imshow(rotated_image)
plt.show()
```

```python
vidcap.set(cv2.CAP_PROP_POS_FRAMES, 1519)
_, f = vidcap.read()
plt.imshow(f)
plt.title(angles[1519, 2])
plt.show()
```

```python
np.load("/Users/gimli/cvr/data/zavity/ETE 2025_07_01-sample/out/Hor_ZH2_down-preprocessed-speeds.npy", allow_pickle=True)
```

```python
np.load("/Users/gimli/cvr/data/zavity/ETE 2025_07_01-sample/out/Hor_ZH2_down-preprocessed-motion_positions.npy")
```

```python

```
