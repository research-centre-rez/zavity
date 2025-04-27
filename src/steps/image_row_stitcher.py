import math
import os
from typing import LiteralString

import cv2
import imageio.v3 as iio
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize
from skimage.metrics import structural_similarity as ssim
from tqdm.auto import tqdm

from config.config import SEARCH_SPACE_SIZE, TESTING_MODE, ROW_ROTATION_OVERLAP_RATIO, XTOL, FTOL, OUTPUT_FOLDER, N_CPUS
from steps.video_camera_motion import VideoMotion


class ImageRowStitcher:
    """
    A class to handle the stitching of image rows extracted from a video.
    This includes aligning rows, correcting shifts, and blending the final output.

    Attributes:
        motions (VideoMotion): Instance of VideoMotion for motion data.
        imageRows (list[np.ndarray]): List of extracted image rows.
        rolledImageRows (list[np.ndarray]): List of image rows after rolling for alignment.
        video_name (str): Name of the video file.
        output_oio_path (str): Full path for saving the final stitched image.
        imgA (np.ndarray): Reference image for alignment.
        imgB (np.ndarray): Target image for alignment.
        seed_position (np.ndarray): Initial seed position for alignment.
        per_row_shift (np.ndarray): Calculated shifts between rows.
        blended_full_image (np.ndarray): Final blended image after stitching.
        physics (dict): Physics parameters for row alignment.
    """
    motions: VideoMotion
    imageRows: list[np.ndarray]
    rolledImageRows: list[np.ndarray]
    video_name: str
    output_oio_path: LiteralString | str | bytes
    physics: dict
    imgA: np.ndarray
    imgB: np.ndarray
    seed_position: np.ndarray
    per_row_shift: np.ndarray
    blended_full_image: np.ndarray

    def __init__(self, rows, motions, video_path):
        """
        Initializes the ImageRowStitcher.

        Args:
            motions (VideoMotion): Instance of VideoMotion for motion data.
            video_path (str): Path to the video file.
        """
        cv2.setNumThreads(N_CPUS)
        self.motions = motions
        self.imageRows = rows
        self.rolledImageRows = []
        self.video_name = os.path.basename(video_path)
        self.output_oio_path = os.path.join(OUTPUT_FOLDER,
                                            os.path.splitext(os.path.basename(video_path))[0] + '-oio.png')
        self.physics = {}

    def process(self):
        """
        Orchestrates the stitching process. If the stitched output already exists,
        skips computation. Otherwise, loads and processes rows.
        """
        if os.path.isfile(self.output_oio_path):
            pass
        else:
            self.align_height()
            self.align_width()
            if not self.motions.is_moving_down():
                self.imageRows.reverse()
            if len(self.imageRows) == 1:
                iio.imwrite(self.output_oio_path, self.imageRows[0].astype(np.uint8))
            else:
                self.load_or_compute()

    def align_height(self):
        """
        Aligns the height of all image rows to the smallest row height.
        Crops rows symmetrically to achieve uniform dimensions.
        """
        shapes = []
        for r in self.imageRows:
            shapes.append(r.shape)
        desired_shape = np.min(shapes, axis=0)
        for i, r in enumerate(self.imageRows):
            crop = r.shape - desired_shape
            self.imageRows[i] = r[math.floor(crop[0] / 2):r.shape[0] - math.ceil(crop[0] / 2), :]

    def align_width(self):
        """
        Aligns the width of all image rows to the median row width.
        Scales each row to achieve uniform dimensions using cubic interpolation.
        """
        shapes = [r.shape for r in self.imageRows]
        widths = [shape[1] for shape in shapes]
        median_width = int(np.median(widths))
        height = shapes[0][0]  # assuming consistent height

        for i, r in enumerate(self.imageRows):
            self.imageRows[i] = cv2.resize(r, (median_width, height), interpolation=cv2.INTER_CUBIC)

    def load_or_compute(self):
        """
        Loads precomputed row positions if available, or computes them.
        Handles rolling and stitching of image rows.
        """
        if os.path.isfile(self._dump_path('positions')):
            self.per_row_shift = np.load(self._dump_path('positions'))
        else:
            self.computePositions()
            np.save(self._dump_path('positions'), self.per_row_shift)

        self.rollImageRows()
        if TESTING_MODE:
            for i, row in enumerate(self.rolledImageRows):
                file_path = os.path.join(OUTPUT_FOLDER, os.path.splitext(self.video_name)[0] + f"-oio-{i}-rolled.png")
                iio.imwrite(file_path, row.astype(np.uint8))
        self.stitchImageRows()
        self.dumpOIO()

    def _dump_path(self, object_name):
        """
        Generates a path for saving or loading an object.

        Args:
            object_name (str): Name of the object.

        Returns:
            str: Path to the file.
        """
        return os.path.join(OUTPUT_FOLDER, os.path.splitext(self.video_name)[0] + f'-{object_name}.npy')

    def dump(self, name: str, object):
        """
        Saves an object to a .npy file.

        Args:
            name (str): Name of the object.
            object: Object to save.
        """
        np.save(self._dump_path(name), object)

    def dumpOIO(self):
        """
        Saves the final blended stitched image to the output path.
        """
        iio.imwrite(self.output_oio_path, self.blended_full_image.astype(np.uint8))

    @staticmethod
    def mutual_information(imgA, imgB, bins=15):
        """
        Computes the mutual information between two images.

        Args:
            imgA (np.ndarray): First image.
            imgB (np.ndarray): Second image.
            bins (int): Number of bins for the histogram.

        Returns:
            float: Mutual information value.
        """
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

    @staticmethod
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
            print(seed_position, shift, x.shape, y.shape, imgB.shape)

            raise Exception

        return -ssim(
            imgA[
                seed_position[0, 0] + SEARCH_SPACE_SIZE[0]: seed_position[0, 0] + SEARCH_SPACE_SIZE[0] + height,
                seed_position[0, 1] + SEARCH_SPACE_SIZE[1]: seed_position[0, 1] + SEARCH_SPACE_SIZE[1] + width
            ].T,
            imgB_interpolated,
            data_range=255
        )

    def to_minimize(self, x):
        """
        Objective function for optimization, minimizes mutual information.

        Args:
            x (tuple): Shift values.

        Returns:
            float: Negative mutual information for given shift.
        """
        return self.extract_images_and_compute_score(
            shift=x,
            imgA=self.imgA,
            imgB=self.imgB,
            seed_position=self.seed_position,
            height=self.imgA.shape[0] - abs(self.physics["shift"]) - 2 * SEARCH_SPACE_SIZE[0],
            width=self.imgA.shape[1] - abs(self.physics["roll"]) - 2 * SEARCH_SPACE_SIZE[1]
        )

    def computePositions(self):
        """
        Computes relative shifts between image rows based on mutual information alignment.
        """
        # TODO: Find out why is it consistently '-20'
        self.physics["shift"] = int(self.motions.get_average_vertical_shift() - 20)

        if ((self.motions.get_direction() == 'CCW' and self.motions.is_moving_down()) or
                (self.motions.get_direction() == 'CW' and not self.motions.is_moving_down())):
            self.physics["roll"] = -int(np.round(self.motions.get_average_horizontal_shift() - self.imageRows[0].shape[1]))
            self.physics["first_frame"] = (self.physics["shift"], 0)
        elif ((self.motions.get_direction() == 'CCW' and not self.motions.is_moving_down()) or
              (self.motions.get_direction() == 'CW' and self.motions.is_moving_down())):
            self.physics["roll"] = int(np.round(self.motions.get_average_horizontal_shift() - self.imageRows[0].shape[1])) + 50
            self.physics["first_frame"] = (self.physics["shift"], self.physics["roll"])

        print(f"Physics: {self.physics}\n")

        first_frame = self.physics["first_frame"]
        scan_shift = self.physics["shift"]
        roll = self.physics["roll"]

        shift_fixes = []
        shift_seeds = []
        score_gains = []

        for i in tqdm(range(len(self.imageRows) - 1), total=(len(self.imageRows) - 1),
                      desc="Computing positions for stitching"):
            self.imgA = self.imageRows[i]
            self.imgB = self.imageRows[i + 1]
            self.seed_position = np.array([first_frame, [first_frame[0] - scan_shift, first_frame[1] - roll]]).astype(int)
            shift_seeds.append(self.seed_position)
            if np.any(np.isnan(self.imgA)) or np.any(np.isnan(self.imgB)):
                print("One of the input images contains NaN values.")
            initial_score = -self.to_minimize((0, 0))
            result = minimize(self.to_minimize, x0=np.array([0, 0]), method='Powell',
                              bounds=[(-SEARCH_SPACE_SIZE[0], SEARCH_SPACE_SIZE[0]),
                                      (-SEARCH_SPACE_SIZE[1], SEARCH_SPACE_SIZE[1])],
                              options={'xtol': XTOL, 'ftol': FTOL, 'disp': TESTING_MODE})
            shift_fixes.append(result.x)

            optimized_score = -result.fun
            score_gain = (optimized_score / initial_score - 1) * 100
            score_gains.append(score_gain)

        self.per_row_shift = np.array([seed[0, :] - seed[1, :] - fix for seed, fix in zip(shift_seeds, shift_fixes)])

        if TESTING_MODE:
            print(f"Score Gain mean: {np.mean(score_gains)}±{np.std(score_gains)}")

        print(f"Calculated: per row shift\n"
              f"{self.per_row_shift}")

    @staticmethod
    def real_roll(array, shift, axis=0):
        """
        Rolls an array circularly along the specified axis, ensuring that the data wraps seamlessly.

        Args:
            array (np.ndarray): Input array to roll.
            shift (int): Amount to shift the array. Positive values shift right, negative values shift left.
            axis (int): Axis along which to roll the array.

        Returns:
            np.ndarray: Rolled array.
        """
        double_image = np.concatenate([array, array], axis=1)
        interp = RegularGridInterpolator(
            (np.arange(double_image.shape[0]), np.arange(double_image.shape[1])),
            double_image
        )
        if shift > 0:
            y = np.arange(array.shape[1] - shift, 2 * array.shape[1] - shift - 0.5)
        else:
            y = np.arange(-shift, array.shape[1] - shift - 0.5)
        x = np.arange(array.shape[0])
        xg, yg = np.meshgrid(x, y)
        return interp((xg, yg)).T

    @staticmethod
    def signed_mod(val, mod_base):
        return ((val + mod_base // 2) % mod_base) - mod_base // 2

    def rollImageRows(self):
        """
        Applies calculated shifts to align image rows horizontally.
        """
        self.rolledImageRows.append(self.imageRows[0])
        for en, row_shift in tqdm(enumerate(np.cumsum(self.per_row_shift[:, 1])), total=self.per_row_shift.shape[0],
                                  desc="Rolling image for stitching"):
            self.rolledImageRows.append(
                self.real_roll(self.imageRows[en + 1], self.signed_mod(row_shift, self.imageRows[en + 1].shape[1])))

    def stitchImageRows(self):
        """
        Stitches the rolled image rows into a single cohesive image.
        """
        to_grid = [self.rolledImageRows[0]]
        real_shift = [0]
        for image, shift in tqdm(zip(self.rolledImageRows[1:], np.cumsum(self.per_row_shift[:, 0])),
                                 total=self.per_row_shift.shape[0], desc="Stitching image"):
            if TESTING_MODE:
                image[0, :] = 0
                image[-1, :] = 0
                image[:, 0] = 0
                image[:, -1] = 0

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
        if TESTING_MODE:
            full_image = np.zeros((out_height, self.rolledImageRows[0].shape[1]))
            for image, r_shift in zip(to_grid, real_shift):
                full_image[r_shift: r_shift + image.shape[0], :] = image
            self.blended_full_image = full_image
            return
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
            blend_matrix[real_shift[en] + to_grid[en].shape[0]: real_shift[en + 2], :, en + 1] += 1
            lin_blend = np.dot(np.linspace(0, 1, (to_grid[en + 1].shape[0] + real_shift[en + 1] - real_shift[en + 2]),
                                           endpoint=False).reshape(-1, 1), np.ones((to_grid[en + 1].shape[1], 1)).T)
            blend_matrix[real_shift[en + 2]: to_grid[en + 1].shape[0] + real_shift[en + 1], :, en + 1] += np.flipud(
                lin_blend)
            blend_matrix[real_shift[en + 2]: to_grid[en + 1].shape[0] + real_shift[en + 1], :, en + 2] += lin_blend

        print("Finishing up...")
        blend_matrix[to_grid[en + 1].shape[0] + real_shift[en + 1]:, :, -1] += 1

        self.blended_full_image = np.sum(full_image * blend_matrix, axis=2) / np.sum(blend_matrix, axis=2)
