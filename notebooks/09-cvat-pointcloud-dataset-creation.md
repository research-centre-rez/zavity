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
import subprocess
from copy import deepcopy

import open3d
import os
from tqdm.auto import tqdm
import pandas as pd
import xml.etree.ElementTree as ET
import shutil
```

```python
def parse_defects(df):
    # Parse defects
    defects = {}
    current_defect = None
    defects_start_found = False
    for i in range(len(df)):
        row = df.iloc[i]
        if row[2] not in ["Nominal", "Nom."] and not defects_start_found:
            continue

        defects_start_found = True
        if row[1] == "X":
            current_defect = row[0]
            defects[current_defect] = {}
            defects[current_defect]["depth"] = row[3]
        if row[1] in ["X", "Y", "Z"]:
            defects[current_defect][row[1]] = row[2]
    return defects
```

```python
def translate_defects(defects, centroid):
    translated_defects = {}
    for key, defect in defects.items():
        translated_defect = deepcopy(defect)
        translated_defect["X"] -= centroid[0]
        translated_defect["Y"] -= centroid[1]
        translated_defect["Z"] -= centroid[2]
        translated_defects[key] = translated_defect
    return translated_defects
```

```python
def write_tracklets(defects, folder):
    # Create KITTI-style XML
    tracklets = ET.Element("tracklets", version="0", tracking_level="0", class_id="0")
    ET.SubElement(tracklets, "count").text = str(len(defects))
    ET.SubElement(tracklets, "item_version").text = "1"

    for defect in defects:
        item = ET.SubElement(tracklets, "item", version="1", tracking_level="0", class_id="1")
        ET.SubElement(item, "objectType").text = "defekt"
        ET.SubElement(item, "h").text = "2"
        ET.SubElement(item, "w").text = "2"
        ET.SubElement(item, "l").text = "2"
        ET.SubElement(item, "first_frame").text = "0"

        poses = ET.SubElement(item, "poses", version="0", tracking_level="0", class_id="2")
        ET.SubElement(poses, "count").text = "1"
        ET.SubElement(poses, "item_version").text = "0"

        pose_item = ET.SubElement(poses, "item", version="1", tracking_level="0", class_id="3")
        ET.SubElement(pose_item, "tx").text = str(defects[defect].get("X", 0))
        ET.SubElement(pose_item, "ty").text = str(defects[defect].get("Y", 0))
        ET.SubElement(pose_item, "tz").text = str(defects[defect].get("Z", 0))
        for axis in ["rx", "ry", "rz"]:
            ET.SubElement(pose_item, axis).text = "0.0"
        for field in ["state", "occlusion", "occlusion_kf", "truncation"]:
            ET.SubElement(pose_item, field).text = "0"
        for amt in ["amt_occlusion", "amt_border_l", "amt_border_r", "amt_occlusion_kf", "amt_border_kf"]:
            ET.SubElement(pose_item, amt).text = "-1"

        ET.SubElement(item, "finished").text = "1"

    # Save the XML
    tree = ET.ElementTree(tracklets)
    tree.write(os.path.join(folder, "converted_tracklets.xml"), encoding="utf-8", xml_declaration=True)
```

```python
ROOT = "/Users/gimli/cvr/data/zavity/EDU 2025_04_24/"
SRC = os.path.join(ROOT, "Repliky ZH EDU03 kolektor")
```

```python
edu_data = os.listdir(SRC)
```

```python
edu_data
```

```python
for folder in tqdm(edu_data):
    if os.path.isdir(os.path.join(SRC, folder)):
        example_stl = os.path.join(SRC, folder, f"replika č.{folder}.stl")
        if os.path.isfile(example_stl):
            mesh = open3d.io.read_triangle_mesh(example_stl)
            mesh.compute_vertex_normals()
            pcd = mesh.sample_points_uniformly(number_of_points=5000000)
            centroid = pcd.get_center()
            pcd.translate(-centroid)
            os.makedirs(os.path.join(ROOT, "cvat-import-multiple", folder, "pointcloud"), exist_ok=True)
            open3d.io.write_point_cloud(os.path.join(ROOT, "cvat-import-multiple", folder, "pointcloud", f"{folder}.pcd"), pcd)

            # Load the Excel file
            df = pd.read_excel(os.path.join(SRC, folder, f"vady č.{folder}.xlsx"), header=None)
            # parse defect from the dataframe
            defects = parse_defects(df)
            defects = translate_defects(defects, centroid)
            # write boxes into annotation format
            os.makedirs(os.path.join(ROOT, "cvat-import-multiple", folder, "annotations", f"{folder}_pcd"), exist_ok=True)
            write_tracklets(defects, os.path.join(ROOT, "cvat-import-multiple", folder, "annotations", f"{folder}_pcd"))

            related_dir = os.path.join(ROOT, "cvat-import-multiple", folder, "related_images", f"{folder}_pcd")
            os.makedirs(related_dir, exist_ok=True)
            for part in ["01", "02"]:
                oio = os.path.join(ROOT, "sken-splitted-outputs", f"hnizdo-{int(folder):02d}-part-{part}-oio.png")
                if os.path.isfile(oio):
                    shutil.copy(oio, os.path.join(related_dir, f"hnizdo-{int(folder):02d}-part-{part}-oio.png") )
```

```python
#example_stl = "/Users/gimli/cvr/data/zavity/EDU 2025_04_24/Repliky ZH EDU03 kolektor/2/replika č.2.stl"
example_stl = "/Users/gimli/cvr/data/zavity/3D_skeny/H_718/Hnizdo_718.stl"
mesh = open3d.io.read_triangle_mesh(example_stl)
mesh.compute_vertex_normals()
h718 = mesh.sample_points_uniformly(number_of_points=5000000)
obbH718 = h718.get_oriented_bounding_box()
```

```python
open3d.visualization.draw_geometries([pcd], window_name="Sampled Point Cloud")
```

```python
open3d.io.write_point_cloud("/Users/gimli/cvr/data/zavity/EDU 2025_04_24/cvat-import-multiple/2/pointcloud/2.pcd", pcd, write_ascii=True)
```

```python
example_stl = "/Users/gimli/cvr/data/zavity/EDU 2025_04_24/Repliky ZH EDU03 kolektor/2/reference č.2.stl"
mesh = open3d.io.read_triangle_mesh(example_stl)
mesh.compute_vertex_normals()
edu2 = mesh.sample_points_uniformly(number_of_points=5000000)
obbEdu2 = edu2.get_oriented_bounding_box()
```

```python
obbH718, obbEdu2
```

```python
edu2.translate(-edu2.get_center())
```

```python
obbEdu2 = edu2.get_oriented_bounding_box()
```

```python
obbEdu2
```

```python
files = os.listdir("/Users/gimli/cvr/data/zavity/EDU 2025_04_24/sken-splitted-outputs")
```

```python
files
```

```python
import numpy as np
import subprocess
```

```python
os.makedirs("/Users/gimli/cvr/data/zavity/EDU 2025_04_24/cvat-oios-import/")
```

```python
filename_parts = [file.split("-") for file in sorted(files) if file.endswith("png")]
hnizda = np.unique([part[1] for part in filename_parts])
for hnizdo in hnizda:
    parts = np.unique([part[3] for part in filename_parts if part[1] == hnizdo])
    for part in parts:
        images = ["-".join(fpart) for fpart in filename_parts if fpart[1] == hnizdo and fpart[3] == part]
        #print(["zip", f"{hnizdo}-{part}.zip"] + [os.path.join(ROOT, "sken-splitted-outputs", image) for image in images])
        subprocess.Popen(["zip", f"/Users/gimli/cvr/data/zavity/EDU 2025_04_24/cvat-oios-import/{hnizdo}-{part}.zip"] + [os.path.join(ROOT, "sken-splitted-outputs", image) for image in images])
```

```python
images = ["-".join(fpart) for fpart in filename_parts if part[1] == hnizdo and fpart[3] == part]
```

```python
part
```

```python
fpart
```

```python

```
