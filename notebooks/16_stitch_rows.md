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
import os
import re
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Rectangle
from skimage.measure import ransac
from skimage.metrics import structural_similarity as ssim
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import differential_evolution
from tqdm.auto import tqdm
import cv2
from matplotlib.patches import Circle
import pandas as pd
from skimage.transform import AffineTransform, ProjectiveTransform, SimilarityTransform
from skimage.morphology import label
from skimage.measure import regionprops
from scipy.ndimage import median_filter

ROOT = "/Users/gimli/cvr/data/zavity/ETE 2025_07_01-sample/out/"
```

```python
imageRows = []
frame_maps = []
for img_path in sorted([os.path.join(ROOT, file) for file in os.listdir(ROOT) if re.match(r'.*oio-[0-9].png', file)]):
    imageRows.append(iio.imread(img_path))
    frame_maps.append(pd.read_csv(img_path.replace(".png", "-frame-map.csv")).to_numpy().reshape(-1).astype(int))
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
top_centers = [(center[0], center[1] + PATCH_SIZE[0] // 2, center[2]) for center in top_centers]

plt.figure(figsize=(15, 3))
ax = plt.subplot(111)
ax.imshow(imageRows[0], cmap="gray")
ax.scatter(p0[0][:, 0, 0], p0[0][:, 0, 1], c='r', s=1)
for center in top_centers:
    ax.add_patch(Circle(center[:2], 100, color='g', fill=False))
plt.show()
```

```python
PATCH_SIZE
```

```python
def create_patch(patch_center, image):
    # wrapper for the constants (PATCH_HEIGHT, PATCH_WIDTH)
    if patch_center[0] < PATCH_SIZE[1] // 2:
        return np.roll(image, axis=1, shift=PATCH_SIZE[1] // 2)[
            int(patch_center[1] - PATCH_SIZE[0] // 2): int(patch_center[1] + PATCH_SIZE[0] // 2),
            int(patch_center[0]): int(patch_center[0] + PATCH_SIZE[1])
        ]
    elif patch_center[0] > image.shape[1] - PATCH_SIZE[1] // 2:
        return np.roll(image, axis=1, shift=-PATCH_SIZE[1] // 2)[
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
def find_patch(image, patch, search_start, search_direction, rough_position=None, ranges=(40, 40), downsample_factor=DOWNSAMPLE_FACTOR):
    """
    Uses MI to detect a similar patch. Because it should not converge, the brute method is used for searching.

    Ideas:
    - is it possible to use a higher downsample factor for a blurred image?
    """
    SEARCH_RANGE = 0.1  # this is ratio of image width in which structural similarity is searched
    patch_similarity_rough = []
    if rough_position is None:
        if search_direction > 0:
            search_range = np.roll(
                np.arange(image.shape[1]),
                axis=0,
                shift=-np.round((search_start - SEARCH_RANGE) * image.shape[1]).astype(int)
            )[:int(SEARCH_RANGE * image.shape[1]):downsample_factor]
        else:
            search_range = np.roll(
                np.arange(image.shape[1], 0, -1) - 1,
                axis=0,
                shift=-image.shape[1] * (1 - search_start)
            )[:int(SEARCH_RANGE * image.shape[1]):downsample_factor]
            print(f"searching from {search_range[0]} to {search_range[-1]}")

        for x in tqdm(search_range,
                      total=search_range.shape[0],
                      desc="Rough patch search:"):
            # TODO: we expect top 1000px of the row as unique. This is usable only in case of second row placed above the first ...
            for y in np.arange(1000, image.shape[0] - patch.shape[0] // 2 + 1, downsample_factor):
                floating_patch = create_patch((x, y), imgB)
                patch_similarity_rough.append(((x, y), ssim(floating_patch, patch, data_range=255)))

        rough_position = np.array(sorted(patch_similarity_rough, key=lambda x: x[1])[-1][0])

    patch_similarity_precise = []
    for x in tqdm(np.arange(rough_position[0] - ranges[1] // 2,
                            rough_position[0] + ranges[1] // 2), total=ranges[1], desc="Precise search:"):
        for y in np.arange(rough_position[1] - ranges[0] // 2,
                           rough_position[1] + ranges[0] // 2):
            floating_patch = create_patch((x, y), image)
            if floating_patch.shape == patch.shape:
                patch_similarity_precise.append(((x, y), ssim(floating_patch, patch, data_range=255)))

    precise_position = np.array(sorted(patch_similarity_precise, key=lambda x: x[1])[-1][0])

    return precise_position, patch_similarity_rough, patch_similarity_precise
```

```python
matching_points = []
for x, y, point_count in tqdm(top_centers, total=len(top_centers), desc="Matching patches: "):
    # There is an offset 50 on y-axis (margin for exporting the patch) which should be eliminated here
    center = int(x), int(y)
    search_start = frame_maps[0][center[0]] / np.max(frame_maps[0])
    search_direction = np.sign(frame_maps[0][-1] - frame_maps[0][0]) # positive is left to right, negative is right to left
    patch = create_patch(center, imageRows[0])

    matching_points.append(find_patch(imgB, patch, search_start, search_direction, downsample_factor=5))
```

```python
[pts - np.array(center)[:2] for (pts, _, position_precise), center in zip(matching_points, top_centers)]
```

```python
plt.figure(figsize=(15, 5))
ax = plt.subplot(211)
ax.imshow(imageRows[0], cmap="gray")
for center in top_centers[:len(matching_points)]:
    ax.add_patch(Circle(np.array(center[:2]), 100, color='g', fill=False, linewidth=2))
ax = plt.subplot(212)
ax.imshow(imgB, cmap="gray")
for pts, _, _ in matching_points:
    ax.add_patch(Circle(pts, 100, color='r', fill=False, linewidth=2))
plt.show()
```

```python
matching_points[0][0]
```

```python
for pid in np.arange(len(matching_points)):
    plt.figure(figsize=(15, 5))
    ax = plt.subplot(121)
    ax.imshow(create_patch(top_centers[pid], imageRows[0]), cmap="gray")
    ax = plt.subplot(122)
    ax.imshow(create_patch(matching_points[pid][0], imageRows[1]), cmap="gray")
    plt.title(f"PID: {pid}")
    plt.show()
```

```python
class ScaleShiftTransform(ProjectiveTransform):
    """Scale + translation transform (no rotation, no shear)."""

    def __init__(self, scale=None, translation=None):
        super().__init__()
        if scale is not None or translation is not None:
            self.estimate(
                np.array([[0, 0], [1, 0]]),
                np.array([[0, 0], [scale, 0]]) + (translation if translation is not None else (0, 0))
            )

    def estimate(self, src, dst):
        """Estimate scale and translation from 2D point correspondences."""
        src = np.asarray(src)
        dst = np.asarray(dst)

        if src.shape != dst.shape or src.shape[1] != 2:
            return False

        # Compute scale (average over x and y distances)
        src_center = src.mean(axis=0)
        dst_center = dst.mean(axis=0)

        src_shifted = src - src_center
        dst_shifted = dst - dst_center

        # ratio of distances gives scale
        src_norm = np.linalg.norm(src_shifted)
        dst_norm = np.linalg.norm(dst_shifted)

        if src_norm == 0:
            return False

        scale = dst_norm / src_norm
        translation = dst_center - scale * src_center

        # Build matrix
        self.params = np.array([
            [scale, 0,      translation[0]],
            [0,     scale,  translation[1]],
            [0,     0,      1]
        ])

        return True

```

```python
model, inliers = ransac((np.array([center[:2] for center in top_centers]),
                         np.array([pts for (pts, _, _) in matching_points])), SimilarityTransform, min_samples=7, residual_threshold=10.0)
```

```python
inliers
```

```python
model.params
```

```python
x_shift = -model.params[0, 2]
y_shift = -model.params[1, 2]

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
    transformed[np.round(y_shift).astype(int):np.round(y_shift).astype(int) + scaled_and_rotated.shape[0], :] = scaled_and_rotated
    weight_matrix[np.round(y_shift).astype(int):, :] = 1.0
    transformed[:imageRows[0].shape[0], :imageRows[0].shape[1]] += imageRows[0].astype(np.float32) / 255.0
    weight_matrix[:imageRows[0].shape[0], :imageRows[0].shape[1]] += 1.0
transformed /= weight_matrix
```

```python
# Create colored representation
```

```python
colored = np.zeros((np.round(imageRows[0].shape[0] + np.abs(y_shift)).astype(int), imageRows[0].shape[1], 3))
if y_shift < 0:
    colored[:scaled_and_rotated.shape[0], :, 0] = scaled_and_rotated
    colored[-np.round(y_shift).astype(int): -np.round(y_shift).astype(int) + imageRows[0].shape[0], :imageRows[0].shape[1], 1] = imageRows[0].astype(np.float32) / 255.0
else:
    colored[np.round(y_shift).astype(int):np.round(y_shift).astype(int) + scaled_and_rotated.shape[0], :, 0] = scaled_and_rotated
    colored[:imageRows[0].shape[0], :imageRows[0].shape[1], 1] = imageRows[0].astype(np.float32) / 255.0
```

```python
import skimage
```

```python
plt.figure(figsize=(25, 6))
ax = plt.subplot(111)
#plt.imshow(skimage.util.compare_images(colored[:,:,0], colored[:,:,1], method="checkerboard", n_tiles=(1, 50)), cmap="gray")
#ax.imshow(np.max(colored, axis=2), cmap="gray")
ax.imshow(colored, cmap="gray")
for inid, center in enumerate(top_centers):
    position = center[0], center[1] - y_shift
    ax.add_patch(Circle(np.array(position), 100, color='green' if inliers[inid] else 'red', fill=False, linewidth=2, alpha=0.8))
plt.xlim(7000, 7500)
plt.ylim(1200, 1600)
plt.axvline(7280)
plt.axvline(7350)
plt.axhline(1440)
plt.axhline(1380)
plt.show()
```

```python
overlap_strip_height = imageRows[1].shape[0] + y_shift
```

# Idea of warping

- let's find out corners (Harris) in first image
- try to find corresponding areas in the second image (differential_evolution)
- there is y_shift and x_shift dependent on x coordinate => create chart
- median filter values to eliminate outliers
- TODO: do some warping to match images

```python
IMAGE_ROW_OVERLAP_HEIGHT = 500
```

```python
# try to find out the corners in the first image
plt.figure(figsize=(15, 2))
img = cv2.cornerHarris(imageRows[0], 30, 21, 0.03)
scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(img.reshape(-1,1)).reshape(imageRows[0].shape)
plt.imshow(np.power(scaled, 1/3) > 0.4, cmap="gray")
plt.colorbar()
plt.title("Corners in the first row")
plt.show()
```

```python
# Find out centers of the regions
spots = label(np.power(scaled, 1/3) > 0.4)
regions = regionprops(spots)
centroids = np.array([r.centroid for r in regions if r.centroid[0] < IMAGE_ROW_OVERLAP_HEIGHT]) # coordinate order [y, x]
```

```python
plt.figure(figsize=(15, 5))
plt.imshow(imageRows[0], cmap="gray")
plt.scatter(centroids[:, 1], centroids[:, 0], color="red", marker="+")
plt.title("Centroids of the corner-regions")
plt.show()
```

```python
np.max(centroids[:, 0])
```

```python
# Find out correspondencies (optimization in range)
X_SHIFT_BOUNDS = (-1100, -500)
Y_SHIFT_BOUNDS = (-500 - 300, -500 + 300)
quality = {}
for centroid in tqdm(centroids, total=len(centroids), desc="Looking for correspondencies"):
    fixed_patch = create_patch((centroid[::-1]), imageRows[0])
    if fixed_patch.shape != PATCH_SIZE:
        continue
    def similarity(patch_center):
        if np.sum(np.isnan(patch_center)) != 0:
            return 1
        moving_patch = create_patch(patch_center, imageRows[1])
        if fixed_patch.shape == moving_patch.shape:
            return -ssim(moving_patch, fixed_patch)
        else:
            return 1
    result = differential_evolution(similarity,
                                    [(centroid[1] + X_SHIFT_BOUNDS[0], centroid[1] + X_SHIFT_BOUNDS[1]),
                                     (imageRows[1].shape[0] + centroid[0] + Y_SHIFT_BOUNDS[0], imageRows[1].shape[0] + centroid[0] + Y_SHIFT_BOUNDS[1])],
                                    strategy="rand2bin")
    quality[(centroid[1], centroid[0])] = result.x, result.fun, result
```

## Visualization of matching points

```python
plt.figure(figsize=(15, 8))
ax = plt.subplot(111)
ax.imshow(np.concatenate([imageRows[1], imageRows[0]], axis=0), cmap="gray")
ax.axhline(imageRows[1].shape[0])

y_median = np.median([q[1] - quality[q][0][1] for q in quality.keys()])
x_median = np.median([q[0] - quality[q][0][0] for q in quality.keys()])
# q => x, y
for cid, q in enumerate(list(quality.keys())):
    color = "red"
    if not( 1.02 > (q[0] - quality[q][0][0]) / x_median > 0.98):  # x mismatch
        color = "yellow"
    if not(1.02 > (q[1] - quality[q][0][1]) / y_median > 0.98):
        if color == "yellow":
            color = "green" # both mismatch
        else:
            color = "orange" # y mismatch
    ax.arrow(q[0], # x
             q[1] + imageRows[1].shape[0], # y
             dy=-(q[1] + imageRows[1].shape[0] - quality[q][0][1]),
             dx=-((q[0] - quality[q][0][0]) % imageRows[1].shape[1]), color=color, width=5)
    # ax.add_patch(Rectangle(((q[0] - 1200) % imageRows[1].shape[1], # x
    #                         q[1] + imageRows[1].shape[0] - 600), # y
    #                        width=700, height=400, color="orange", fill=False))
    #ax.scatter([6600], [1050], color="red", s=200)
#plt.xlim(3000, 4000)
# plt.ylim(900, 1750)
plt.title("Correspondencies - red = close to median, yellow = x_median missed, orange = y_median missed, green = both missed")
plt.show()
```

```python
# Histogram of translation values
plt.figure(figsize=(15, 3))
ax = plt.subplot(121)
ax.hist([q[0] - quality[q][0][0] for q in quality.keys()], bins=50)
ax.set_title("Translation X")
ax = plt.subplot(122)
ax.hist([q[1] - quality[q][0][1] for q in quality.keys()], bins=50)
ax.set_title("Translation Y")
plt.show()
```

```python
y_compensation = sorted([(q[0], q[1] - quality[q][0][1]) for q in quality.keys() if y_median * 0.98 > q[1] - quality[q][0][1] > y_median * 1.02], key=lambda x: x[0])
```

```python
data = sorted([(q[0], q[0] - quality[q][0][0], q[1] - quality[q][0][1]) for q in quality.keys()], key=lambda x: x[0])
plt.figure(figsize=(15, 5))
plt.plot([x for x, x_trans, y_trans in data], median_filter([x_trans/x_median for x, x_trans, y_trans in data], 33), label="medfilt(x-translation)")
plt.plot([x for x, x_trans, y_trans in data], median_filter([y_trans/y_median for x, x_trans, y_trans in data], 33), label="medfilt(y-translation)")
plt.axhline(1, color="green", alpha=0.5)
plt.xlabel("x coordinate in image row")
plt.ylabel("distance from median")
plt.legend()
plt.show()
```

```python
import numpy as np
import cv2

def _wrap_diff(dx, W):
    # map differences to (-W/2, W/2]
    return ((dx + W/2) % W) - W/2

def _circular_median(diffs_wrapped):
    # ordinary median on wrapped values in (-W/2, W/2]
    return np.median(diffs_wrapped)

def _mad(a):
    med = np.median(a)
    return np.median(np.abs(a - med)) + 1e-9, med

def _nanmedian_filter_1d(arr, k):
    # odd kernel; nan-safe median filter
    assert k % 2 == 1
    pad = k // 2
    padded = np.pad(arr, (pad, pad), mode='edge')
    out = np.empty_like(arr)
    for i in range(len(arr)):
        window = padded[i:i+k]
        out[i] = np.nanmedian(window)
    return out

def _linear_interp_nan(y):
    # fill NaNs by linear interpolation; endpoints by nearest
    x = np.arange(len(y))
    m = np.isfinite(y)
    if m.sum() == 0:
        return np.zeros_like(y)
    y_filled = y.copy()
    y_filled[~m] = np.interp(x[~m], x[m], y[m])
    # clamp endpoints by nearest
    first, last = np.where(m)[0][[0, -1]]
    y_filled[:first] = y_filled[first]
    y_filled[last+1:] = y_filled[last]
    return y_filled

def _make_bins(x, W, n_bins=256, min_pts=5, values=None, reducer=np.median):
    """
    Bin samples at positions x in [0, W) into n_bins,
    reduce values in each bin by reducer (median), return per-column series via interpolation.
    """
    x = np.asarray(x)
    v = np.asarray(values)
    # bin edges and centers
    edges = np.linspace(0, W, n_bins+1)
    centers = 0.5*(edges[:-1] + edges[1:])
    out = np.full(n_bins, np.nan, dtype=float)

    # assign to bins
    idx = np.clip(np.searchsorted(edges, x, side='right') - 1, 0, n_bins-1)
    print(idx)
    for bin_id in range(n_bins):
        values_to_bin_mask = (idx == bin_id)
        if np.count_nonzero(values_to_bin_mask) >= min_pts:
            out[bin_id] = reducer(v[values_to_bin_mask])

    # interpolate to all columns
    series = np.interp(np.arange(W),
                       np.clip((centers / (W-1))*(W-1), 0, W-1),
                       _linear_interp_nan(out))
    return series

def _pav_isotonic(y):
    """
    Pool-Adjacent-Violators to enforce nondecreasing y.
    O(n) expected, simple implementation.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    g = y.copy()
    w = np.ones(n, dtype=float)
    i = 0
    while i < n-1:
        if g[i] <= g[i+1]:
            i += 1
            continue
        # pool blocks
        j = i
        while j >= 0 and g[j] > g[j+1]:
            # merge j and j+1
            new_w = w[j] + w[j+1]
            new_g = (g[j]*w[j] + g[j+1]*w[j+1]) / new_w
            g[j] = new_g
            w[j] = new_w
            # delete block j+1 by shifting left
            g[j+1:j+ (n - (j+1))] = g[j+2:n]
            w[j+1:j+ (n - (j+1))] = w[j+2:n]
            n -= 1
            j -= 1
            if j < 0:
                break
        # rebuild full-length arrays by piecewise-constant expansion
        # (simplify by re-expanding at the end)
        # To keep it simple, restart
        # Re-expand g,w to length len(y)
        # NOTE: For robustness/simplicity we do a simpler isotonic via cumulative max afterwards.
        break

    # Simple, stable fallback: project by cumulative maximum of the minimum slope 0
    # (monotone nondecreasing, closest in L-infinity sense)
    g = np.maximum.accumulate(y)
    return g

def _project_monotone(xs):
    # ensure strictly nondecreasing; small epsilon to avoid ties
    xm = _pav_isotonic(xs)
    # small tweak to avoid equal neighbors (helps cv2.remap edge cases)
    eps = 1e-6
    diffs = np.maximum(np.diff(xm), 0)
    ties = diffs == 0
    if np.any(ties):
        xm[1:] += np.cumsum(ties) * eps
    return xm

def warp_scan(
    img2,
    matches,           # array Nx4: [x1, y1, x2, y2] in pixel coords (float)
    W, H,
    n_bins=256,
    median_filter_kernel=11,     # odd
    local_win=64,      # columns for local outlier test
    max_mad_x=0.05,
    max_mad_y=0.15,
):
    """
    Constrained warp of img2 into img1 coords using matches and 360° x-axis.

    Returns: warped, mask, y_shift_median, x_roll
    """
    assert median_filter_kernel % 2 == 1
    matches = np.asarray(matches, dtype=float)
    x1, y1, x2, y2 = matches.T

    # 1) robust circular roll on x
    dx_wrap = _wrap_diff(x1 - x2, W)
    # coarse outlier trim for roll
    median_deviation, median_x = _mad(dx_wrap)
    print(f"average diff from median(x)={median_deviation:.3f}, median(x)={median_x:.3f}")
    inl = np.abs(dx_wrap - median_x) <= 0.02 * median_x
    roll = _circular_median(dx_wrap[inl]) if inl.any() else _circular_median(dx_wrap)
    print(f"roll => {roll:.3f}")

    # unwrap x2 near x1 using r
    rolled_x = x2 + roll
    # bring x2r close to x1 by adding multiples of W
    k = np.round((x1 - rolled_x) / W)
    aligned_x = rolled_x + k * W  # unrolled x2 near x1

    # raw diffs
    dx = aligned_x - x1         # desired per-point x offset at x1
    dy = y1 - y2          # desired per-point y shift at x1
    y_shift_median = float(np.median(dy[np.isfinite(dy)]))

    # 2) local-median outlier culling
    order = np.argsort(x1)
    x1_ordered, dx_ordered, dy_ordered = x1[order], dx[order], dy[order]

    def filter_outliers(values, window_size, median_deviation_ratio):
        inlier_mask = np.ones_like(values, dtype=bool)
        half_window_size = max(1, window_size // 2)
        for value_index in range(len(values)):
            left_index = max(0, value_index - half_window_size)
            right_index = min(len(values), value_index + half_window_size + 1)
            windowed_values = values[left_index:right_index]
            window_median = np.median(windowed_values)
            window_std = np.std(windowed_values)
            if np.abs(values[value_index] - window_median) > window_std * median_deviation_ratio:
                inlier_mask[value_index] = False
        return inlier_mask

    kx = filter_outliers(dx_ordered, local_win, max_mad_x)
    ky = filter_outliers(dy_ordered, local_win, max_mad_y)
    inliers_mask = kx & ky
    x1_filtered, dx_filtered, dy_filtered = x1_ordered[inliers_mask], dx_ordered[inliers_mask], dy_ordered[inliers_mask]

    # 3) per-column Δx (piecewise x-stretch), then enforce monotone xs = xd + Δx
    dx_series = _make_bins(x1_filtered, W, n_bins=n_bins, min_pts=5, values=dx_filtered, reducer=np.median)
    plt.figure(figsize=(15, 3))
    plt.plot(dx_series)
    plt.plot(x1_filtered, dx_filtered)
    plt.show()
    dx_series = _nanmedian_filter_1d(dx_series, median_filter_kernel)
    dx_series = _linear_interp_nan(dx_series)  # safety

    xd = np.arange(W, dtype=float)
    xs = xd + dx_series
    xs = _project_monotone(xs)

    # 4) per-column Δy, plus global median
    dy_series = _make_bins(x1_filtered, W, n_bins=n_bins, min_pts=5, values=dy_filtered, reducer=np.median)
    dy_series = _nanmedian_filter_1d(dy_series, median_filter_kernel)
    dy_series = _linear_interp_nan(dy_series)

    # 5) build dense maps and remap
    Himg, Wimg = img2.shape[:2]
    assert Himg == H and Wimg == W, "img2 size must be HxW"

    map_x = np.tile(xs[None, :], (H, 1)).astype(np.float32)
    # y_s = y_d - Δy(x_d)
    map_y = (np.arange(H, dtype=float)[:, None] - dy_series[None, :]).astype(np.float32)

    warped = cv2.remap(
        img2, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # valid mask: remap an all-ones image
    ones = np.ones((H, W), dtype=np.float32)
    valid_f = cv2.remap(
        ones, map_x, map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mask = (valid_f > 0.5).astype(np.uint8)

    return warped, mask, y_shift_median, float(roll)

# -------------------------
# Example usage (pseudo):
#
# warped, mask, y_med, x_roll = warp_scan(img2, matches, W=img2.shape[1], H=img2.shape[0])
# cv2.imwrite("warped.png", warped)
# cv2.imwrite("mask.png", (mask*255))
# print("y_shift_median:", y_med, "   x_roll:", x_roll)

```

```python
matches = np.array([[q[0], q[1], quality[q][0][0], quality[q][0][1]] for q in quality.keys()]).astype(np.float32)
img1 = imageRows[0]
img2 = cv2.resize(imageRows[1], (W, H))
```

```python
warped, mask, y_shift_median, x_roll = warp_scan(img2, matches,  # array Nx4: [x1, y1, x2, y2] in pixel coords (float)
                                                 W, H,
                                                 n_bins=35,
                                                 median_filter_kernel=21,  # odd
                                                 local_win=64,  # columns for local outlier test
                                                 max_mad_x=4.0,
                                                 max_mad_y=6.0)
```

```python
y_shift_median
```

```python
plt.figure(figsize=(15, 5))
ax = plt.subplot(311)
ax.imshow(img2, cmap="gray")
ax = plt.subplot(312)
ax.imshow(warped, cmap="gray")
ax = plt.subplot(313)
ax.imshow(img1, cmap="gray")
plt.show()
```

```python
x1, y1, x2, y2 = [matches[:, i].astype(np.float64) for i in range(4)]

# Sort by x2 so "nearby" really means nearby columns
order = np.argsort(x2)
x1, y1, x2, y2 = x1[order], y1[order], x2[order], y2[order]
```

```python
#plt.plot(x1, x2 - x1)
plt.plot(x1, rolling_median_outlier_mask(x2-x1, 31, 0.3))
plt.show()
```

```python

```
