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
import numpy as np
from tqdm.auto import tqdm
```

```python
f_params = {
      "maxCorners": 200,
      "qualityLevel": 0.001,
      "minDistance": 2,
      "blockSize": 15
}

lk_params = {
    # Size of the search window at each pyramid level
    "winSize": [30, 30],
    # 0-based maximal pyramid level. If set to 0, pyramids are not used (therefore its single level).
    # If set to 1, then two pyramids are used, and so on
    "maxLevel": 4,

    # Param specifying the termination criteria of the iterative search algorithm.
    # Alternatively when the search window moves by less that criteria.epsilon
    # criteria: [type, maxCount, epsilon] where:
        # type: type of termination criteria. 3 => the desired accuracy or change in params at which the iterative algorithm stops
        # maxCount: The maximum number of iterations per point that's being tracked
        # epsilon: The desired accuracy or change in params at which the iterative algorithm stops
    "criteria": [3, 100, 0.01]
}
```

```python
vidcap = cv2.VideoCapture("/Users/gimli/cvr/data/SvornikyZH/TestSken18072025/sken253MMzaMIN.MP4")

success, frame = vidcap.read()
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
corners = cv2.goodFeaturesToTrack(prev_gray, mask=None, **f_params)

chunks = [{
    "frame_start": 0,
    "corners": corners,
    "trajectories": [[corner] for corner in corners],
    "shifts": []
}]

with tqdm(desc="Calculating optical flow", total=int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))) as pbar:
    frame_no = 0
    while True:
        ret, frame = vidcap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        new_corners, status, error = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, corners, None, **lk_params
        )

        if len(new_corners) < 150:
            corners = cv2.goodFeaturesToTrack(prev_gray, mask=None, **f_params)
            new_corners, status, error = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, corners, None, **lk_params
            )
            chunks.append({
                "frame_start": frame_no -  1,
                "corners": corners,
                "trajectories": [[corner] for corner in corners],
                "shifts": []
            })

        mean_error = np.mean(error[status == 1])
        pbar.set_postfix(mean_error=f"{mean_error:.4f}, pts={len(new_corners)}")
        good_new_corners = new_corners[status == 1]

        shifts = []
        for cid, corner in enumerate(new_corners):
            if status[cid] == 1:
                chunks[-1]["trajectories"][cid].append(corner)
                shifts.append(chunks[-1]["trajectories"][cid][-1] - chunks[-1]["trajectories"][cid][-2])
        chunks[-1]["shifts"].append(shifts)

        prev_gray = gray.copy()
        corners = good_new_corners.reshape(-1, 1, 2)

        pbar.update(1)
        frame_no += 1

    pbar.close()
    vidcap.release()
```

```python
import matplotlib.pyplot as plt
```

```python
shift_median = np.array([
    [np.median(np.array(shifts)[:,:,0]), np.median(np.array(shifts)[:,:,1])]
    for chunk in chunks for shifts in chunk["shifts"]
])
plt.scatter(
   shift_median[:,0], shift_median[:,1], alpha=0.1
)
plt.show()
```

```python
plt.plot(shift_median[:, 0])
plt.ylim(-1,1)
plt.show()
```

```python
x_shift = np.copy(shift_median[:, 0])
x_shift[np.abs(x_shift) > 0.5] = 0
```

```python
plt.plot(np.cumsum(shift_median[:, 0][np.abs(shift_median[:, 0]) < 0.5]))
plt.show()
```

```python
y_shift = np.copy(shift_median[:,1])
y_shift[np.abs(shift_median[:,1] - np.median(shift_median[:, 1])) > 2] = np.median(shift_median[:,1])
```

```python
plt.figure(figsize=(15, 5))
plt.plot(shift_median[:, 1])
plt.plot(y_shift)
plt.axvline(2535, color="red")
plt.show()
```

```python
len(shift_median)
```

```python
vidcap = cv2.VideoCapture("/Users/gimli/cvr/data/SvornikyZH/TestSken18072025/sken253MMzaMIN.MP4")

success, frame = vidcap.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
frames_count = 2535
height = np.ceil(frames_count * np.abs(np.median(y_shift))).astype(int)
x_cum = np.cumsum(x_shift)
width = np.ceil(frame.shape[1] + np.max(x_cum) - np.min(x_cum)).astype(int)

x_offset = -np.min(x_cum).astype(int)
accumulator = np.zeros((height, width))
weights = np.zeros((height, width))

y_cum = np.cumsum(y_shift)

if np.ceil(np.median(y_shift)).astype(int) > 0: # bottom-up
    for frame_no in np.arange(frames_count):
        left_top_corner_pos = [
            height - np.round(y_cum[frame_no]).astype(int),
            (x_offset + x_cum[frame_no]).astype(int)
        ]
        if left_top_corner_pos[0] - np.ceil(np.median(y_shift)).astype(int) - 1 < 0:
            break
        accumulator[
            left_top_corner_pos[0] - np.ceil(np.median(y_shift)).astype(int) - 1: left_top_corner_pos[0],
            left_top_corner_pos[1]: left_top_corner_pos[1] + frame.shape[1]
        ] += frame[
            frame.shape[0] // 2 :  frame.shape[0] // 2 + np.ceil(np.median(y_shift)).astype(int) + 1,
            :
        ]
        weights[
            left_top_corner_pos[0] - np.ceil(np.median(y_shift)).astype(int) - 1: left_top_corner_pos[0],
            left_top_corner_pos[1]: left_top_corner_pos[1] + frame.shape[1]
        ] += 1

        success, frame = vidcap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
else: # top-down
    for frame_no in np.arange(frames_count):
        left_top_corner_pos = [
            - np.round(y_cum[frame_no]).astype(int),
            (x_offset + x_cum[frame_no]).astype(int)
        ]
        if left_top_corner_pos[0] - np.ceil(np.median(y_shift)).astype(int) + 2 > height:
            break
        accumulator[
            left_top_corner_pos[0]: left_top_corner_pos[0] - np.ceil(np.median(y_shift)).astype(int) + 2,
            left_top_corner_pos[1]: left_top_corner_pos[1] + frame.shape[1]
        ] += frame[
            frame.shape[0] // 2 :  frame.shape[0] // 2 - np.ceil(np.median(y_shift)).astype(int) + 2,
            :
        ]
        weights[
            left_top_corner_pos[0]: left_top_corner_pos[0] - np.ceil(np.median(y_shift)).astype(int) + 2,
            left_top_corner_pos[1]: left_top_corner_pos[1] + frame.shape[1]
        ] += 1

        success, frame = vidcap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
```

```python
plt.figure(figsize=(15, 50))
plt.imshow((accumulator / weights).astype(np.uint8), cmap="gray")
plt.show()
```

```python
import imageio.v3 as iio
```

```python
iio.imwrite("/Users/gimli/cvr/data/SvornikyZH/TestSken18072025/sken253MMzaMIN-oio.png", (accumulator/weights).astype(np.uint8))
```

```python
accumulator.shape
```

```python

```
