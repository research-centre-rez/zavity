import logging
import math
import os
import cv2
import imageio.v3 as iio
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from tqdm.auto import tqdm

from config.config import (IMAGE_REPEATS, BLENDED_PIXELS_PER_FRAME, BLENDED_PIXELS_SHIFT, OUTPUT_FOLDER, TESTING_MODE, SINUSOID_SAMPLING)
from steps.video_camera_motion import VideoMotion


class ImageRowBuilder:
    frames: np.ndarray
    motions: VideoMotion

    def __init__(self, motions, intervals, video_file_path):
        self.motions = motions
        self.intervals = intervals
        self.video_file_path = video_file_path

        video_capture = cv2.VideoCapture(video_file_path)
        self.width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_capture.release()

    def construct_rows(self):
        """
        Constructs image rows by processing video frames.
        """
        logging.info(f"Processing RowBuilder for: {self.video_file_path}\n")
        rows = []
        rows_compensated = []
        for iid, interval in enumerate(self.intervals):
            int_start, int_end = interval
            start = int_start + (int_end - int_start) // 2 - self.motions.get_frames_per360() // 2
            end = start + self.motions.get_frames_per360()
            file_path = os.path.join(OUTPUT_FOLDER,
                                     os.path.splitext(os.path.basename(self.video_file_path))[0] + f"-oio-{iid}.png")
            if not os.path.isfile(file_path):
                row = self.construct_row(int(start), int(end), iid)
                rows.append(row)
                row_compensated = self.remove_column_shifts(row, self.motions.motion_positions[int(start): int(end), 2])
                rows_compensated.append(row_compensated)
            else:
                row = iio.imread(file_path)
                rows.append(row)

        rows_sin_compensated = self.remove_sin_transform(rows)

        if TESTING_MODE:
            for iid, row in enumerate(rows):
                file_path = os.path.join(OUTPUT_FOLDER,
                                         os.path.splitext(os.path.basename(self.video_file_path))[0] + f"-oio-{iid}.png")
                iio.imwrite(file_path, row.astype(np.uint8))

        if TESTING_MODE:
            for iid, row in enumerate(rows_compensated):
                file_path = os.path.join(OUTPUT_FOLDER,
                                         os.path.splitext(os.path.basename(self.video_file_path))[0] + f"-oio-{iid}-compensated.png")
                iio.imwrite(file_path, row.astype(np.uint8))

        if TESTING_MODE:
            for iid, row in enumerate(rows_sin_compensated):
                file_path = os.path.join(OUTPUT_FOLDER,
                                         os.path.splitext(os.path.basename(self.video_file_path))[0] + f"-oio-{iid}-sin.png")
                iio.imwrite(file_path, row.astype(np.uint8))

        return rows

    def construct_row(self,
                      start: int,
                      end: int,
                      row_id: int,
                      blended_pixels_per_frame=BLENDED_PIXELS_PER_FRAME,
                      blended_pixels_shift=BLENDED_PIXELS_SHIFT):
        """
        Constructs a single row image from video frames.

        Args:
            start (int): Starting frame index.
            end (int): Ending frame index.
            row_id (int): Order of the row.
            blended_pixels_per_frame (int): Pixels blended per frame.
            blended_pixels_shift (int): Shift for blending.

        Returns:
            np.ndarray: Constructed row image.
        """
        if end - start <= 0:
            raise IOError(f"Sequence has negative number of frames {start}-{end}.")

        frame_size = (self.width, self.height)

        shift_per_frame = self.motions.get_horizontal_speed()
        frames_per_360_deg = self.motions.get_frames_per360()
        direction = self.motions.get_direction()

        image_part = self.height

        offset = max(0, math.ceil(start - (blended_pixels_per_frame // 2) / shift_per_frame))
        n_frames = math.ceil(frames_per_360_deg + (blended_pixels_per_frame - 1) / shift_per_frame)

        frame_shift_to_pixels_total = math.ceil(n_frames * shift_per_frame) + (blended_pixels_per_frame - 1) * 2
        row_image = np.zeros(
            (frame_size[0],
             frame_shift_to_pixels_total))

        weight_matrix = np.zeros(row_image.shape)

        vidcap = cv2.VideoCapture(self.video_file_path)
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, offset)
        for frameNo in tqdm(range(0, n_frames), desc=f"Building row image {row_id}"):
            success, frame = vidcap.read() # shape (h, w), grayscale
            if not success:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            shift = (row_image.shape[1] - shift_per_frame * frameNo - blended_pixels_per_frame) if direction == "CCW" else (
                    shift_per_frame * frameNo)
            shift_partial = shift % 1
            shift_matrix = np.float32([
                [1, 0, shift_partial],
                [0, 1, 0]
            ])

            aligned_image = cv2.warpAffine(image, shift_matrix, (frame_size[1] + 1, frame_size[0]))

            crop_x_start = math.floor(shift)
            crop_x_end = max(0, crop_x_start + blended_pixels_per_frame)

            # Add the cropped aligned image slice to row_image and weight matrix
            try:
                row_image[:, crop_x_start:crop_x_end] += aligned_image[:, (image_part // 2 - blended_pixels_per_frame // 2) + blended_pixels_shift:
                                                            (image_part // 2 + 1 + blended_pixels_per_frame // 2) + blended_pixels_shift]
            except:
                raise Exception(f"Row builder failed on adding slice to row_image.\n"
                                f"Row image shape{row_image.shape}\n"
                                f"Crop from {crop_x_start} to {crop_x_end}\n"
                                f"Adding image of shape {aligned_image.shape}\n"
                                f"Offset {offset}\n"
                                f"Frame {frameNo}")
            weight_matrix[:, crop_x_start:crop_x_end] += 1

        # Normalize and crop borders
        row_size = math.floor(frames_per_360_deg * shift_per_frame)
        start_col = (row_image.shape[1] - row_size) // 2
        end_col = start_col + row_size
        row_image = (row_image / weight_matrix)[:, start_col:end_col]

        return np.copy(row_image)

    def remove_sin_transform(self, rows):
        """
        Wrapper function to remove sinusoidal distortions.

        Args:
            rows (list[np.ndarray]): List of image rows.

        Returns:
            list[np.ndarray]: Corrected image rows.
        """
        if rows:
            movementses = []
            for iid, interval in enumerate(self.intervals):
                int_start, int_end = interval
                start = int_start + (int_end - int_start) // 2 - self.motions.get_frames_per360() // 2
                end = start + self.motions.get_frames_per360()
                movementses.append(self.motions.motion_positions[int(start): int(end), 2] - (np.max(self.motions.motion_positions[int(start): int(end), 2]) + np.min(self.motions.motion_positions[int(start): int(end), 2])) / 2)
            movementses = self.detrend_movements(movementses)
            params = self.fitSin(movementses)
            logging.debug(f"\nParameters for sinusoidal transformation:\n{params}\n")
            rows = self.remove_sinusoidal_transformation(rows, params)

        return rows

    def detrend_movements(self, movementses):
        """
        Removes linear trends from cumulative movements.

        Args:
            movementses (list[np.ndarray]): List of cumulative movements.

        Returns:
            list[np.ndarray]: List of detrended cumulative movements.
        """
        detrended_movementses = []
        for i, movements in enumerate(movementses):
            x = np.arange(len(movements))
            coefficients = np.polyfit(x, movements, deg=1)
            linear_trend = np.polyval(coefficients[-2:], x)
            detrended_movementses.append(movements - linear_trend)

        return detrended_movementses

    def fitSin(self, movementses):
        """
        Fits a rotated sinusoidal model to cumulative movements.

        Args:
            movementses (list[np.ndarray]): List of cumulative movements.

        Returns:
            np.ndarray: Fitted sinusoidal parameters.
        """
        paramses = []
        for i, movements in enumerate(movementses):
            x = np.arange(len(movements))
            movements = np.asarray(movements)

            # Remove NaN or inf values
            mask = np.isfinite(movements)
            if not np.any(mask):
                raise ValueError(f"All movement values are NaN/inf for index {i}")
            x = x[mask]
            movements = movements[mask]

            max_m = np.max(movements)
            min_m = np.min(movements)
            freq = 2 * np.pi / len(movements)

            custom_rotated_sinusoid = lambda x, A, C, D, theta: self.rotated_sinusoid(x, A, freq, C, D, theta)
            initial_guesses = [(max_m - min_m) / 2, 0, 0, 0]
            lower_bounds = [0, -np.pi, -100, -0.1]
            upper_bounds = [max(max_m, -min_m), np.pi, +100, 0.1]

            try:
                params, pcov = curve_fit(custom_rotated_sinusoid, x, movements, p0=initial_guesses,
                                      bounds=(lower_bounds, upper_bounds), method='trf', maxfev=5000)
            except RuntimeError as e:
                logging.critical(f"Fit failed for index {i}: {e}")
                continue  # or fill with default values if needed

            A, C, D, theta = params
            paramses.append((A, freq, C, D, theta))

        return np.array(paramses)

    def remove_sinusoidal_transformation(self, images, paramses):
        """
        Removes sinusoidal distortions from image rows.

        Args:
            images (list[np.ndarray]): List of image rows.
            paramses (np.ndarray): Sinusoidal parameters for correction.

        Returns:
            list[np.ndarray]: Corrected image rows.
        """
        rows = []
        # A, B, _, _, _ = np.median(paramses, axis=0)
        for image, params in zip(images, paramses):
            A, B, C, _, _ = params

            # Create an empty output image
            output_image = np.zeros_like(image)

            # Track the maximum shift
            max_shift = 0

            # Loop over each column
            for j in range(image.shape[1]):
                # Calculate the vertical shift for this column based on the sinusoidal function
                shift = self.sinusoid(j, A, B / SINUSOID_SAMPLING, C, 0)
                max_shift = max(max_shift, abs(shift))  # Update maximum shift

                # Shift the whole column
                # Use np.roll to shift the column by the calculated value
                new_column = np.roll(image[:, j], int(shift))

                # Assign the shifted column back to the output image
                output_image[:, j] = new_column

            # Crop the image to remove the wrapped-around pixels
            if max_shift > 0:
                output_image = output_image[int(max_shift): -int(max_shift), :]

            rows.append(output_image)

        return rows

    @staticmethod
    def sinusoid(x, A, B, C, D):
        """
        Defines a sinusoidal function.

        Args:
            x (np.ndarray | float): Input values.
            A (float): Amplitude.
            B (float): Frequency.
            C (float): Phase shift.
            D (float): Vertical shift.

        Returns:
            np.ndarray | float: Sinusoidal output.
        """
        return A * np.sin(B * x + C) + D

    @staticmethod
    def rotated_sinusoid(x, A, B, C, D, theta):
        """
        Rotates a sinusoidal function by a given angle.

        Args:
            x (np.ndarray | float): Input values.
            A, B, C, D (float): Sinusoidal parameters.
            theta (float): Rotation angle.

        Returns:
            np.ndarray | float: Rotated sinusoidal output.
        """
        y = A * np.sin(B * x + C) + D
        x_rot = x * np.cos(theta) - y * np.sin(theta)
        y_rot = x * np.sin(theta) + y * np.cos(theta)
        return y_rot

    def remove_column_shifts(self, image, shifts):
        """
        Removes vertical shifts from a row

        Args:
            image (np.ndarray): created rows of the threaded socket
            shifts (np.ndarray): Measured corrections

        Returns:
            np.ndarray: Corrected image rows.
        """

        # Create an empty output image
        output_image = np.zeros_like(image)

        # Track the maximum shift
        max_shift = 0

        shift_func = interp1d(np.linspace(0, 1, len(shifts)), shifts - (np.max(shifts) + np.min(shifts)) / 2)
        stretched_shifts = -shift_func(np.linspace(0, 1, image.shape[1]))

        # Loop over each column
        for shift, j in zip(stretched_shifts, np.arange(image.shape[1])):
            max_shift = max(max_shift, abs(shift))  # Update maximum shift

            # Shift the whole column
            # Use np.roll to shift the column by the calculated value
            new_column = np.roll(image[:, j], int(shift))

            # Assign the shifted column back to the output image
            output_image[:, j] = new_column

        # Crop the image to remove the wrapped-around pixels
        if max_shift > 0:
            output_image = output_image[int(max_shift): -int(max_shift), :]

        return output_image
