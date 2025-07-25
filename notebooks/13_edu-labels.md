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

- z EDU jsme dostali v SVG označená místa, která bychom měli highlightnout v rámci zobrazení
- cílem je přečíst data z SVG
- zkusit nějaký thresholding nebo segmentační síť (podle toho, kolik bude dat)

```python
ROOT = "/Users/gimli/cvr/data/zavity/EDU 2025_04_24"
labels_dir = "Hodnocení"
oios_dir = "sken-splitted-outputs"
filename = "hnizdo-02-part-02-oio"
```

```python
import os
import xml.etree.ElementTree as ET
import imageio.v3 as iio
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import base64
from io import BytesIO
```

```python
svgs = [os.path.join(ROOT, labels_dir, file) for file in os.listdir(os.path.join(ROOT, labels_dir)) if file.endswith("oio.svg")]
```

```python
def parse_svg_ellipses(svg):
    tree = ET.parse(svg)
    root = tree.getroot()

    # SVG namespace (may vary based on your SVG)
    ns = {
        'svg': 'http://www.w3.org/2000/svg',
        'xlink': 'http://www.w3.org/1999/xlink'
    }

    image = None
    ellipses = []

    elements = list(root.iter())
    # NOTE: it is not clear if all ellipses should be used (in some cases there are object invisible - background)
    for el in elements:
        if el.tag.endswith('image'):
            img_data = el.get("{http://www.w3.org/1999/xlink}href")
            if img_data.startswith('data:image'):
                base64_string = img_data.split(',')[1]
                image_bytes = base64.b64decode(base64_string)
                with BytesIO(image_bytes) as f:
                    image = iio.imread(f)
            x = float(el.attrib.get("x"))
            y = float(el.attrib.get("y"))
            width = float(el.attrib.get("width"))
            height = float(el.attrib.get("height"))
        elif image is not None and el.tag.endswith('ellipse'):
            ellipse = el
            cx_raw = float(ellipse.attrib.get('cx', 0))
            cy_raw = float(ellipse.attrib.get('cy', 0))
            rx_raw = float(ellipse.attrib.get('rx', 0))
            ry_raw = float(ellipse.attrib.get('ry', 0))

            tform_str = ellipse.attrib.get("transform", None)
            if tform_str != None:
                if tform_str.startswith("matrix"):
                    a, b, c, d, e, f = [float(num) for num in tform_str.split("(")[1].split(")")[0].split(",")]
                    cx = cx_raw * a + cy_raw * c + e - x
                    cy = cx_raw * b + cy_raw * d + f - y
                    rx = np.sqrt(np.power(a * rx_raw, 2) + np.power(b * rx_raw, 2))
                    ry = np.sqrt(np.power(c * ry_raw, 2) + np.power(d * ry_raw, 2))
                    angle = np.arctan2(b, a)
                elif tform_str.startswith("rotate"):
                    angle = np.deg2rad([float(num) for num in tform_str.split("(")[1].split(")")[0].split(",")][0])
                    cx = cx_raw * np.cos(angle) - cy_raw * np.sin(angle) - x
                    cy = cx_raw * np.sin(angle) + cy_raw * np.cos(angle) - y
                    rx = rx_raw
                    ry = ry_raw
            else:
                cx = cx_raw - x
                cy = cy_raw - y
                rx = rx_raw
                ry = ry_raw
                angle = 0


            ellipses.append((cx, cy, rx * 2, ry * 2, angle))
    return cv2.resize(image, (int(width), int(height))), ellipses
```

```python
def draw_svg(svg):
    image, ellipses = parse_svg_ellipses(svg)
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(15, 10))
    plt.imshow(image, cmap="gray")
    for center in ellipses:
        ellipse = Ellipse((center[0], center[1]), width=center[2], height=center[3], edgecolor='orange', facecolor='none')
        ellipse.set_angle(np.rad2deg(center[4]))
        ax.add_patch(ellipse)

    plt.scatter([ellipse[0] for ellipse in ellipses], [ellipse[1] for ellipse in ellipses], marker="+", color="red", linewidths=1)
    plt.title(os.path.basename(svg))
    plt.show()
```

```python
for svg in sorted(svgs):
    draw_svg(svg)
```

```python

```
