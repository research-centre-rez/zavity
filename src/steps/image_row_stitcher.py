import math
import os
import re
from typing import LiteralString

import imageio.v3 as iio
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize
from tqdm.auto import tqdm

from config import SEARCH_SPACE_SIZE
from steps.video_camera_motion import VideoMotion


class ImageRowStitcher:
    motions: VideoMotion
    imageRows: list[np.ndarray]
    rolledImageRows: list[np.ndarray]
    video_name: str
    output_path: str
    output_oio_path: LiteralString | str | bytes
    video_name: str
    imgA: np.ndarray
    imgB: np.ndarray
    seed_position: np.ndarray
    per_row_shift: np.ndarray
    blended_full_image: np.ndarray
    physics: dict

    def __init__(self, output_path, motions, video_path):
        self.motions = motions
        self.imageRows = []
        self.rolledImageRows = []
        self.video_name = os.path.basename(video_path)
        self.output_path = output_path
        self.output_oio_path = os.path.join(output_path, os.path.splitext(os.path.basename(video_path))[0] + '-oio.png')

    def process(self):
        if os.path.isfile(self.output_oio_path):
            pass
        else:
            self.load_img_rows()
            self.alighHeight()
            if len(self.imageRows) == 1:
                iio.imwrite(self.output_oio_path, self.imageRows[0].astype(np.uint8))
            else:
                self.load_or_compute()

    def load_img_rows(self):
        pattern = rf"^{re.escape(os.path.splitext(self.video_name)[0])}.*-oio-.\.png$"
        for filename in [file for file in os.listdir(self.output_path) if re.match(pattern, file)]:
            self.imageRows.append(iio.imread(os.path.join(self.output_path, filename)))
        print(f"Loaded {len(self.imageRows)} row images\n")

    def alighHeight(self):
        shapes = []
        for r in self.imageRows:
            shapes.append(r.shape)
        desired_shape = np.min(shapes, axis=0)
        for i, r in enumerate(self.imageRows):
            crop = r.shape - desired_shape
            self.imageRows[i] = r[math.floor(crop[0] / 2):r.shape[0] - math.ceil(crop[0] / 2),
                             math.floor(crop[1] / 2):r.shape[1] - math.ceil(crop[1] / 2)]

    def _dump_path(self, object_name):
        return os.path.join(self.output_path, os.path.splitext(self.video_name)[0] + f'-{object_name}.npy')

    def dump(self, name: str, object):
        np.save(self._dump_path(name), object)

    def load_or_compute(self):
        if os.path.isfile(self._dump_path('positions.npy')):
            self.per_row_shift = np.load(self._dump_path('positions.npy'))
        else:
            self.computePositions()
            np.save(self._dump_path('positions.npy'), self.per_row_shift)

        self.rollImageRows()
        self.stitchImageRows()
        self.dumpOIO()

    def dumpOIO(self):
        iio.imwrite(self.output_oio_path, self.blended_full_image.astype(np.uint8))

    @staticmethod
    def cropArr(arr):
        return arr[:, np.any((arr != 0), axis=0)]

    # def loadImageRows(self):
    #     for filename in sorted(os.listdir(self.src), reverse=not self.move_down):
    #         if self.video_name.removesuffix(".MP4") + "-oio-" in filename and "png" in filename:
    #             self.imageRows.append(self.cropArr(iio.imread(os.path.join(self.src, filename))))

    @staticmethod
    def mutual_information(imgA, imgB, bins=15):
        # taken from https://matthew-brett.github.io/teaching/mutual_information.html
        hist_2d, x_edges, y_edges = np.histogram2d(
            imgA.ravel(),
            imgB.ravel(),
            bins=bins
        )
        pxy = hist_2d / float(np.sum(hist_2d))
        px = np.sum(pxy, axis=1)  # marginal for x over y
        py = np.sum(pxy, axis=0)  # marginal for y over x
        px_py = px[:, None] * py[None, :]  # Broadcast to multiply marginals

        # Now we can do the calculation using the pxy, px_py 2D arrays
        nzs = pxy > 0  # Only non-zero pxy values contribute to the sum
        return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

    def extract_images_and_compute_mi(self, shift, imgA, imgB, seed_position, width, height):
        # print(shift, height, width, seed_position)
        x = np.arange(seed_position[1, 0] + shift[0], seed_position[1, 0] + shift[0] + height - 0.5)
        y = np.arange(seed_position[1, 1] + shift[1], seed_position[1, 1] + shift[1] + width - 0.5)
        # print(x.shape, y.shape)
        xg, yg = np.meshgrid(x, y)
        # print(xg[0, 0], xg[-1,-1], yg[0,0], yg[-1,-1], imgA.shape, imgB.shape)
        # print(imgA[seed_position[0, 0]: seed_position[0, 0] + 600,
        #     seed_position[0, 1]: seed_position[0, 1] + width].T.shape)
        interp = RegularGridInterpolator((np.arange(imgB.shape[0]), np.arange(imgB.shape[1])), imgB)
        return -self.mutual_information(
            imgA[seed_position[0, 0]: seed_position[0, 0] + height,
            seed_position[0, 1]: seed_position[0, 1] + width].T,
            interp((xg, yg))
        )

    def to_minimize(self, x):
        return self.extract_images_and_compute_mi(shift=x, imgA=self.imgA, imgB=self.imgB,
                                                  seed_position=self.seed_position,
                                                  width=self.imgA.shape[1] - 2 * abs(self.physics["roll"]) -
                                                        2 * SEARCH_SPACE_SIZE[0],
                                                  height=self.imgA.shape[0] - self.physics["scan_shift"] -
                                                         2 * SEARCH_SPACE_SIZE[1])

    @staticmethod
    def show_images(imgA, imgB, seed_position, shift, width):
        plt.figure(figsize=(8, 40))
        ax = plt.subplot(131)
        ax.imshow(imgA[seed_position[0, 0]: seed_position[0, 0] + 600,
                  seed_position[0, 1]: seed_position[0, 1] + width].T, cmap="gray")
        ax.set_title("Fixed")

        x = np.arange(seed_position[1, 0] + shift[0], seed_position[1, 0] + shift[0] + 599.5)
        y = np.arange(seed_position[1, 1] + shift[1], seed_position[1, 1] + shift[1] + width - 0.5)
        xg, yg = np.meshgrid(x, y)
        interp = RegularGridInterpolator((np.arange(imgB.shape[0]), np.arange(imgB.shape[1])), imgB)

        ax = plt.subplot(132)
        ax.imshow(interp((xg, yg)), cmap="gray")
        ax.set_title("Moved")

        ax = plt.subplot(133)
        ax.imshow(imgA[seed_position[0, 0]: seed_position[0, 0] + 600,
                  seed_position[0, 1]: seed_position[0, 1] + width].T + interp((xg, yg)), cmap="gray")
        ax.set_title("Blend")
        plt.show()

    def computePositions(self):
        self.physics = {
            "scan_shift": int(self.motions.getAverageVerticalShift()),
        }

        if self.motions.getDirection() == 'CCW':
            self.physics["roll"] = int(np.round(self.imageRows[0].shape[1] * 0.05))
        else:
            self.physics["roll"] = -int(np.round(self.imageRows[0].shape[1] * 0.05))

        self.physics["first_frame"] = (self.physics["scan_shift"] + SEARCH_SPACE_SIZE[1],
                                       abs(self.physics["roll"]) + SEARCH_SPACE_SIZE[0])

        print(f"Physics: {self.physics}\n")

        first_frame = self.physics["first_frame"]
        scan_shift = self.physics["scan_shift"]
        roll = self.physics["roll"]

        shift_fixes = []
        shift_seeds = []

        for i in tqdm(range(len(self.imageRows) - 1), total=(len(self.imageRows) - 1),
                      desc="Computing positions for stitching"):
            self.imgA = self.imageRows[i]
            self.imgB = self.imageRows[i + 1]
            # print(self.imgA.shape, self.imgB.shape)
            self.seed_position = np.array([first_frame, [first_frame[0] - scan_shift, first_frame[1] + roll]]).astype(
                int)
            shift_seeds.append(self.seed_position)
            if np.any(np.isnan(self.imgA)) or np.any(np.isnan(self.imgB)):
                print("One of the input images contains NaN values.")
            result = minimize(self.to_minimize, x0=np.array([0, 0]), method='Powell',
                              bounds=[(-SEARCH_SPACE_SIZE[1], +SEARCH_SPACE_SIZE[1]),
                                      (-SEARCH_SPACE_SIZE[0], SEARCH_SPACE_SIZE[0])],
                              options={'xtol': 1e-2, 'ftol': 1e-2})
            shift_fixes.append(result.x)

        self.per_row_shift = np.array([seed[0, :] - seed[1, :] - fix for seed, fix in zip(shift_seeds, shift_fixes)])

        print(f"Calculated: per row shift\n"
              f"{self.per_row_shift}")

    @staticmethod
    def real_roll(array, shift, axis=0):
        double_image = np.concatenate([array, array], axis=1)
        interp = RegularGridInterpolator(
            (np.arange(double_image.shape[0]), np.arange(double_image.shape[1])),
            double_image
        )
        if shift > 0:
            y = np.arange(shift, shift + array.shape[1] - 0.5)
        else:
            y = np.arange(shift + array.shape[1], 2 * array.shape[1] + shift - 0.5)
        x = np.arange(array.shape[0])
        xg, yg = np.meshgrid(x, y)
        return interp((xg, yg)).T

    def rollImageRows(self):
        self.rolledImageRows.append(self.imageRows[0])
        for en, row_shift in tqdm(enumerate(np.cumsum(-self.per_row_shift[:, 1])), total=self.per_row_shift.shape[0],
                                  desc="Rolling image for stitching"):
            self.rolledImageRows.append(
                self.real_roll(self.imageRows[en + 1], row_shift % self.imageRows[en + 1].shape[1]))

    def stitchImageRows(self):
        to_grid = [self.rolledImageRows[0]]
        real_shift = [0]
        for image, shift in tqdm(zip(self.rolledImageRows[1:], np.cumsum(self.per_row_shift[:, 0])),
                                 total=self.per_row_shift.shape[0], desc="Stitching image"):
            interp = RegularGridInterpolator(
                (np.arange(image.shape[0]), np.arange(image.shape[1])),
                image
            )
            x = np.arange(image.shape[0] - 1)
            y = np.arange(image.shape[1])
            xg, yg = np.meshgrid(x, y)
            to_grid.append(interp((xg, yg)).T)
            real_shift.append(int(shift // 1))

        out_height = self.rolledImageRows[0].shape[0] + np.max(real_shift)
        full_image = np.zeros((out_height, self.rolledImageRows[0].shape[1], len(self.rolledImageRows)))
        for en, (image, r_shift) in enumerate(zip(to_grid, real_shift)):
            full_image[r_shift: r_shift + image.shape[0], :, en] = image

        blend_matrix = np.zeros((out_height, to_grid[0].shape[1], len(to_grid)))

        blend_matrix[:real_shift[1], :, 0] += 1
        lin_blend = np.dot(np.linspace(0, 1, (to_grid[0].shape[0] - real_shift[1]), endpoint=False).reshape(-1, 1),
                           np.ones((to_grid[0].shape[1], 1)).T)
        blend_matrix[real_shift[1]: to_grid[0].shape[0], :, 0] += np.flipud(lin_blend)
        blend_matrix[real_shift[1]: to_grid[0].shape[0], :, 1] += lin_blend

        for en, (image, r_shift) in tqdm(enumerate(list(zip(to_grid, real_shift))[:-2]),
                                         total=len(list(zip(to_grid, real_shift))[:-2]), desc="Blending image"):
            # print(en, real_shift[en + 2], to_grid[en + 1].shape[0], real_shift[en + 1],(to_grid[en + 1].shape[0] +
            # real_shift[en + 1] - real_shift[en + 2]), np.linspace(0, 1, 1 / (to_grid[en + 1].shape[0] + real_shift[
            # en + 1] - real_shift[en + 2])).shape, np.ones((to_grid[en + 1].shape[1], 1)).T.shape)
            blend_matrix[real_shift[en] + to_grid[en].shape[0]: real_shift[en + 2], :, en + 1] += 1
            lin_blend = np.dot(np.linspace(0, 1, (to_grid[en + 1].shape[0] + real_shift[en + 1] - real_shift[en + 2]),
                                           endpoint=False).reshape(-1, 1), np.ones((to_grid[en + 1].shape[1], 1)).T)
            blend_matrix[real_shift[en + 2]: to_grid[en + 1].shape[0] + real_shift[en + 1], :, en + 1] += np.flipud(
                lin_blend)
            blend_matrix[real_shift[en + 2]: to_grid[en + 1].shape[0] + real_shift[en + 1], :, en + 2] += lin_blend

        print("Finishing up...")
        blend_matrix[to_grid[en + 1].shape[0] + real_shift[en + 1]:, :, -1] += 1

        self.blended_full_image = np.sum(full_image * blend_matrix, axis=2) / np.sum(blend_matrix, axis=2)

