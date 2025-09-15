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
import os
import re
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm.auto import tqdm
import cv2
from matplotlib.patches import Circle

ROOT = "/Users/gimli/cvr/data/zavity/ETE 2025_07_01-sample/out/"
```

```python
imageRows = []
for img_path in sorted([os.path.join(ROOT, file) for file in os.listdir(ROOT) if re.match(r'.*oio-[0-9].png', file)]):
    imageRows.append(iio.imread(img_path))
```

```python
# align height
shapes = []
for r in imageRows:
    shapes.append(r.shape)
desired_shape = np.min(shapes, axis=0)
```

```python
%matplotlib inline
plt.figure(figsize=(15, 3))
ax = plt.subplot(2, 1, 1)
ax.imshow(imageRows[0][:400, 4000:9000])
ax = plt.subplot(2, 1, 2)
ax.imshow(imageRows[1][-520:-120, 3250:8150])
plt.show()
```

```python
imageRows[0].shape
```

```python
def select_patch(points, R, tol=1e-9):
    """
    Return (center_x, center_y), max_count for a circle of radius R
    that covers the most points.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        raise ValueError("No points.")
    R2 = R * R

    def count_at(c):
        d2 = np.sum((pts - c)**2, axis=1)
        return int(np.count_nonzero(d2 <= R2 + tol))

    best_c = pts[0]
    best_n = count_at(best_c)

    # Try each point as a center (handles tiny-R / sparse cases)
    for p in pts:
        n = count_at(p)
        if n > best_n:
            best_n, best_c = n, p

    # Try centers defined by each pair (<= 2R apart)
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            pi, pj = pts[i], pts[j]
            dvec = pj - pi
            d = float(np.hypot(dvec[0], dvec[1]))
            if d < tol or d > 2 * R + tol:
                continue

            mid = (pi + pj) * 0.5
            h2 = R2 - (d * 0.5) ** 2
            if h2 < -1e-12:
                continue
            h = 0.0 if h2 < 0 else float(np.sqrt(max(0.0, h2)))

            # Perpendicular unit vector to (pj - pi)
            ux, uy = dvec / d
            perp = np.array([-uy, ux])

            for c in (mid + h * perp, mid - h * perp):
                n = count_at(c)
                if n > best_n:
                    best_n, best_c = n, c

    return (float(best_c[0]), float(best_c[1])), best_n

```

```python
import numpy as np

def top_dense_centers(points, R, k=5, separation=None):
    """
    Return up to k centers (chosen from the input points) that each cover many
    neighbors within radius R, spaced at least `separation` apart.

    Parameters
    ----------
    points : iterable of (x, y)
    R : float
        Radius used to count neighbors.
    k : int, default 5
        Max number of centers to return.
    separation : float or None, default None
        Minimum spacing between returned centers. Defaults to R.

    Returns
    -------
    centers : list of (x, y, count)
        Each tuple is the chosen center (from the input points) and how many
        points are within distance <= R of it.
    """
    P = np.asarray(points, dtype=float)
    n = len(P)
    if n == 0:
        return []

    if separation is None:
        separation = R

    # Pairwise squared distances (n x n)
    D2 = np.sum((P[:, None, :] - P[None, :, :])**2, axis=2)

    # How many neighbors within R for each point-centered circle
    counts = (D2 <= R * R).sum(axis=1)

    # Greedy non-maximum suppression for uniformity
    order = np.argsort(-counts)          # best first
    sep2 = separation * separation
    chosen = []
    chosen_idx = []

    for i in order:
        if all(D2[i, j] >= sep2 for j in chosen_idx):
            chosen_idx.append(i)
            chosen.append((float(P[i, 0]), float(P[i, 1]), int(counts[i])))
            if len(chosen) >= k:
                break

    return chosen

```

```python
PATCH_SIZE = (100, 500)  # height, width

p0 = cv2.goodFeaturesToTrackWithQuality(cv2.GaussianBlur(imageRows[0][PATCH_SIZE[0] // 2:400], (31, 1), 11), maxCorners=500, qualityLevel=0.1, minDistance=25, mask=None, blockSize=40)
top_centers = top_dense_centers(p0[0].reshape(-1,2), 100, k=10, separation=300)

plt.figure(figsize=(15, 3))
ax = plt.subplot(111)
ax.imshow(imageRows[0][PATCH_SIZE[0] // 2:400])
ax.scatter(p0[0][:, 0, 0], p0[0][:, 0, 1], c='r', s=1)
for center in top_centers:
    ax.add_patch(Circle(center, 100, color='g', fill=False))
plt.show()
```

```python
def create_patch(patch_center, image):
    # wrapper for the constants (PATCH_HEIGHT, PATCH_WIDTH)
    if patch_center[0] < PATCH_SIZE[1] // 2:
        return np.roll(image, axis=0, shift=PATCH_SIZE[1] // 2)[
            int(patch_center[1] - PATCH_SIZE[0] // 2): int(patch_center[1] + PATCH_SIZE[0] // 2),
            int(patch_center[0]): int(patch_center[0] + PATCH_SIZE[1])
        ]
    elif patch_center[0] > image.shape[1] - PATCH_SIZE[1] // 2:
        return np.roll(image, axis=0, shift=-PATCH_SIZE[1] // 2)[
            int(patch_center[1] - PATCH_SIZE[0] // 2): int(patch_center[1] + PATCH_SIZE[0] // 2),
            int(patch_center[0] - PATCH_SIZE[1]): int(patch_center[0])
        ]
    else:
        return image[int(patch_center[1] - PATCH_SIZE[0] // 2): int(patch_center[1] + PATCH_SIZE[0] // 2),
                     int(patch_center[0] - PATCH_SIZE[1] // 2): int(patch_center[0] + PATCH_SIZE[1] // 2)]
```

```python
imgB = imageRows[1]
```

```python
DOWNSAMPLE_FACTOR = 10
def find_patch(image, patch, rough_position=None, ranges=(40, 40), downsample_factor=DOWNSAMPLE_FACTOR):
    """
    Uses MI to detect a similar patch. Because it should not converge, the brute method is used for searching.

    Ideas:
    - is it possible to use higher downsample factor for blurred image?
    """
    patch_similarity_rough = None
    if rough_position is None:
        patch_similarity_rough = np.zeros((image.shape[0] // downsample_factor, image.shape[1] // downsample_factor))
        for x in tqdm(np.arange(patch.shape[1] // 2, image.shape[1] - patch.shape[1] // 2 + 1, downsample_factor),
                      total=(image.shape[1] - patch.shape[1]) // downsample_factor,
                      desc="Rough patch search:"):
            for y in np.arange(1000, image.shape[0] - patch.shape[0] // 2 + 1, downsample_factor):
                floating_patch = create_patch((x, y), imgB)
                patch_similarity_rough[y // downsample_factor, x // downsample_factor] = ssim(floating_patch, patch, data_range=255)

        rough_position = np.array(np.unravel_index(np.argmax(patch_similarity_rough), shape=patch_similarity_rough.shape)) * downsample_factor

    patch_similarity_precise = np.zeros(ranges)
    for enx, x in tqdm(enumerate(np.arange(rough_position[1] - ranges[1] // 2,
                                           rough_position[1] + ranges[1] // 2)), total=ranges[1], desc="Precise search:"):
        for eny, y in enumerate(np.arange(rough_position[0] - ranges[0] // 2,
                                          rough_position[0] + ranges[0] // 2)):
            floating_patch = create_patch((x, y), image)
            if floating_patch.shape == patch.shape:
                patch_similarity_precise[eny, enx] = ssim(floating_patch, patch, data_range=255)

    precise_position_delta = np.unravel_index(np.argmax(patch_similarity_precise), shape=patch_similarity_precise.shape)

    precise_position = rough_position + np.array(ranges) // 2 + np.array(precise_position_delta)

    return precise_position, patch_similarity_rough, patch_similarity_precise
```

```python
matching_points = []
for x, y, point_count in tqdm(top_centers, total=len(top_centers), desc="Matching patches: "):
    # There is an offset 50 on y-axis (margin for exporting the patch) which should be eliminated here
    center = int(x), int(y + 50)
    patch = create_patch(center, imageRows[0])
    matching_points.append(find_patch(imgB, patch, downsample_factor=5))
```

```python
[pts - np.array(center)[:2][::-1] for (pts, _, _), center in zip(matching_points, top_centers)]
```

```python
plt.figure(figsize=(15, 5))
ax = plt.subplot(211)
ax.imshow(imageRows[0], cmap="gray")
for center in top_centers[:len(matching_points)]:
    ax.add_patch(Circle(np.array(center[:2]) + np.array((0, 50)), 100, color='g', fill=False, linewidth=2))
ax = plt.subplot(212)
ax.imshow(imgB, cmap="gray")
for pts, _, _ in matching_points:
    ax.add_patch(Circle((pts[1], pts[0] - 35), 100, color='r', fill=False, linewidth=2))
ax.axhline(1100, color="blue")
plt.show()
```

```python
model, inliers = ransac((np.array([center[:2][::-1] for center in top_centers]),
                         np.array([pts for (pts, _, _) in matching_points])), SimilarityTransform, min_samples=5, residual_threshold=5.0)
```

```python
x_shift = model.params[0, 2]
y_shift = model.params[1, 2]

affine = np.eye(3)
affine[:2, :2] = model.params[:2, :2]

rolled = np.roll(imgB, np.round(x_shift).astype(int), axis=1)
scaled_and_rotated = cv2.warpAffine(rolled.astype(np.float32)/255.0, affine[:2].astype(np.float32), (rolled.shape[1], rolled.shape[0]))
transformed = np.zeros((np.round(imageRows[0].shape[0] + np.abs(y_shift)).astype(int), imageRows[0].shape[1]))
weight_matrix = np.zeros((np.round(imageRows[0].shape[0] + np.abs(y_shift)).astype(int), imageRows[0].shape[1]))
if y_shift < 0:
    transformed[:scaled_and_rotated.shape[0], :] = scaled_and_rotated
    weight_matrix[:scaled_and_rotated.shape[0], :] = 1.0
    transformed[-np.round(y_shift).astype(int): -np.round(y_shift).astype(int) + imageRows[0].shape[0], :imageRows[0].shape[1]] += imageRows[0].astype(np.float32) / 255.0
    weight_matrix[-np.round(y_shift).astype(int): -np.round(y_shift).astype(int) + imageRows[0].shape[0], :imageRows[0].shape[1]] += 1.0
else:
    transformed[np.round(y_shift).astype(int):, :] = scaled_and_rotated
    weight_matrix[np.round(y_shift).astype(int):, :] = 1.0
    transformed[:imageRows[0].shape[0], :imageRows[0].shape[1]] += imageRows[0].astype(np.float32) / 255.0
    weight_matrix[:imageRows[0].shape[0], :imageRows[0].shape[1]] += 1.0
transformed /= weight_matrix
```

```python
plt.figure(figsize=(25, 6))
plt.imshow(transformed, cmap="gray")
plt.xlim(6000, 8000)
plt.show()
```

```python
from skimage import transform

warped = transform.warp(imgB.astype(np.float32) / 255.0, model, output_shape=(imageRows[0].shape[0] * 2, imageRows[0].shape[1]))
warped[-imageRows[0].shape[0]:, :] += imageRows[0].astype(np.float32) / 255.0
```

```python
plt.figure(figsize=(25, 6))
plt.imshow(warped, cmap="gray")
plt.show()
```

```python
coords, pr, pp = find_patch(imgB, patch, ranges=(20, 20))
```

```python
varx = []
for x in np.arange(-1000, 3000, 20):
    patch2_center = (int(patch_center[0] - imageRows[0].shape[1] // 2 + x), int(patch_center[1]))
    varx.append(np.std(create_patch(patch2_center, imageRows[0]).reshape(-1)))
```

```python
patch2_center = (int(patch_center[0] - imageRows[0].shape[1] // 2 + np.arange(-1000, 3000, 20)[np.argmax(varx)]), int(patch_center[1]))
```

```python
plt.plot(varx)
plt.show()
```

```python
patch2 = create_patch(patch2_center, imageRows[0])
```

```python
coords2, pr2, pp2 = find_patch(imgB, patch2)
```

```python
coords2[::-1] - patch2_center, coords[::-1] - patch_center
```

```python
plt.imshow(patch2, cmap="gray")
plt.show()
```

```python
plt.imshow(create_patch(coords2[::-1], imgB), cmap="gray")
plt.show()
```

```python
plt.figure(figsize=(15, 10))
ax = plt.subplot(211)
ax.imshow(imageRows[0], cmap="gray")
ax.add_patch(Circle(patch_center, 100, color='g', fill=False, linewidth=2))
ax.add_patch(Circle(patch2_center, 100, color='r', fill=False, linewidth=2))
ax = plt.subplot(212)
ax.imshow(imgB, cmap="gray")
ax.add_patch(Circle((coords[1], coords[0]), 100, color='g', fill=False, linewidth=2))
ax.add_patch(Circle((coords2[1], coords2[0]), 100, color='r', fill=False, linewidth=2))
plt.show()
```

```python
vary = []
for y in np.arange(-100, 300, 20):
    patch2_center = (int(patch_center[0] - imageRows[0].shape[1] // 2), int(patch_center[1] + y))
    vary.append(np.std(create_patch(patch2_center, imageRows[0]).reshape(-1)))

patch3_center = (int(patch_center[0] - imageRows[0].shape[1] // 2), int(patch_center[1] + np.arange(-100, 300, 20)[np.argmax(vary)]))
```

```python
patch3 = create_patch(patch3_center, imageRows[0])
```

```python
plt.imshow(patch3, cmap="gray")
plt.show()
```

```python
coords3, pr3, pp3 = find_patch(imgB, patch3)
```

```python
plt.imshow(create_patch(coords3[::-1], imgB), cmap="gray")
plt.show()
```

```python
ssim(patch3, create_patch(coords3[::-1], imgB))
```

```python
coords3[::-1], patch3_center
```

```python
plt.imshow(pp3, cmap="gray")
plt.show()
```

```python
%%time
simm = []
for dx in tqdm(np.arange(0, 2000, 100)):
    sim = []
    for dy in np.arange(1, 1000, 50):
        sim.append(ssim(imageRows[0][:400, 4000:9000],
                        imageRows[1][-400 - dy: -dy, 4000 + dx:9000 + dx], data_range=255))
    simm.append(sim)
```

```python
%matplotlib notebook
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
x = np.arange(0, 2000, 100)
y = np.arange(1, 1000, 50)
X, Y = np.meshgrid(x, y)
R = np.array(simm).reshape(len(x), len(y))
ax.plot_wireframe(X, Y, R)
plt.show()
```

```python
speeds = np.load(os.path.join(ROOT, "Hor_ZH2_down-preprocessed-speeds.npy"), allow_pickle=True)
```

```python
speeds
```

```python
intervals = np.load(os.path.join(ROOT, "Hor_ZH2_down-preprocessed-intervals.npy"), allow_pickle=True)
```

Je tu nějaká mess ohledně nastavení proměnných níže, podle toho jestli jde o CW, CCW, top-down nebo bottom-up scanning.
CCW down = CW up, CCW up = CW down
- shift je nastaven vždy stejně
- roll otáčí znaménko a příčítá 50
- first frame je (shift, 0) nebo (shift, roll)

```python
scan_shift = int(speeds["vertical_shift"] - 20)
roll = int(np.round(speeds["horizontal"] * np.mean(intervals[:, 1] - intervals[:, 0]) + imageRows[0].shape[1]))
first_frame = (speeds["vertical_shift"] - 20, 0)
```

```python
np.mean(intervals[:, 1] - intervals[:, 0]) * speeds["horizontal"]
```

```python
from steps.image_row_stitcher import ImageRowStitcher
```

```python
imgA = imageRows[-1]
imgB = imageRows[-2]
```

```python
seed_position = np.array([first_frame, [first_frame[0] - scan_shift, imgA.shape[1] + roll]]).astype(int)
```

```python
seed_position
```

```python
width = imgA.shape[0] - np.abs(scan_shift) - 2 * 24
height = imgA.shape[1] - 2 * 36
```

```python
width, height
```

```python
from config.config import SEARCH_SPACE_SIZE
from scipy.interpolate import RegularGridInterpolator

```

```python
def extract_images_and_compute_score(shift, imgA, imgB, seed_position, width, height):
    """
    Extracts image regions and computes mutual information.

    Args:
        shift (tuple): Shift applied to the image.
        imgA (np.ndarray): Fixed image.
        imgB (np.ndarray): Moving image.
        seed_position (np.ndarray): Seed position for alignment.
        width (int): Width of the region.
        height (int): Height of the region.

    Returns:
        float: Negative mutual information.
    """
    x = np.arange(
        seed_position[1, 0] + SEARCH_SPACE_SIZE[0] + shift[0],
        seed_position[1, 0] + SEARCH_SPACE_SIZE[0] + shift[0] + height - 0.5
    )
    y = np.arange(
        seed_position[1, 1] + SEARCH_SPACE_SIZE[1] + shift[1],
        seed_position[1, 1] + SEARCH_SPACE_SIZE[1] + shift[1] + width - 0.5
    )
    xg, yg = np.meshgrid(x, y)
    interp = RegularGridInterpolator((np.arange(imgB.shape[0]), np.arange(imgB.shape[1])), imgB)
    try:
        imgB_interpolated = interp((xg, yg))
    except Exception:
        logging.critical(
            f"Interpolation during score computing went wrong\n"
            f"Seed position: {seed_position}\n"
            f"Shift: {shift}\n"
            f"ImgB shape: {imgB.shape}"
        )

        raise Exception(
            f"Interpolation during score computing went wrong\n"
            f"Seed position: {seed_position}\n"
            f"Shift: {shift}\n"
            f"ImgB shape: {imgB.shape}"
        )

    return -ssim(
        imgA[
            seed_position[0, 0] + SEARCH_SPACE_SIZE[0]: seed_position[0, 0] + SEARCH_SPACE_SIZE[0] + height,
            seed_position[0, 1] + SEARCH_SPACE_SIZE[1]: seed_position[0, 1] + SEARCH_SPACE_SIZE[1] + width
        ].T,
        imgB_interpolated,
        data_range=255
    )
```

```python
interp = RegularGridInterpolator(points=(np.arange(imgB.shape[0]), np.arange(imgB.shape[1])), values=imgB)
```

```python
x = np.arange(
        seed_position[1, 0] + SEARCH_SPACE_SIZE[0] + scan_shift,
        seed_position[1, 0] + SEARCH_SPACE_SIZE[0] + scan_shift + height - 0.5
    )
```

```python
y = np.arange(
        seed_position[1, 1] + SEARCH_SPACE_SIZE[1] + 0,
        seed_position[1, 1] + SEARCH_SPACE_SIZE[1] + 0 + width - 0.5
    )
```

```python
xg, yg = np.meshgrid(x, y)
```

```python
np.max(xg), np.min(xg), imgB.shape
```

```python
imgB_interpolated = interp((xg, yg))
```

```python
ImageRowStitcher.extract_images_and_compute_score((0, 0), imgA, imgB, seed_position,
                                                  imgA.shape[0] - np.abs(scan_shift) - 2 * 24,
                                                  imgA.shape[1] - 2 * 36)
```

```python

```
