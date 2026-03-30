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
TARGET = "/Volumes/FUEL_ZH_TEAM/ETE 2026_03_16/stripe-videos"
```

```python
import os
import pandas as pd
import shutil
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
                rot_videos.append(os.path.join(ROOT, file))

    return video_filename, sorted(rot_videos)
```

```python
for ROOT, rotate in configurations:
    video_filename, rot_videos = load_and_sort_videos(ROOT)
    for video in rot_videos:
        if not os.path.exists(os.path.join(TARGET, os.path.basename(video))):
            print(video, os.path.basename(video))
            shutil.copy(video, os.path.join(TARGET, os.path.basename(video)))
```

```python
TARGET = "/Volumes/FUEL_ZH_TEAM/ETE 2026_03_16/colormaps"
for ROOT, rotate in configurations:
    for file in os.listdir(ROOT):
        if file.endswith("diverging-colormap.png"):
            print(file)
            shutil.copy(os.path.join(ROOT, file), os.path.join(TARGET, file))
```

```python

```
