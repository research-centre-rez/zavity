import logging
import math
import os
import cv2
import imageio.v3 as iio
import numpy as np
from matplotlib import pyplot as plt, patches
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from sklearn.metrics import r2_score
from tqdm.auto import tqdm

from config.config import (N_CPUS, VERBOSE, BLENDED_PIXELS_PER_FRAME, BLENDED_PIXELS_SHIFT, SINUSOID_SAMPLING,
                           IMAGE_REPEATS, LOAD_VIDEO_TO_RAM, OUTPUT_FOLDER, TESTING_MODE, STRIPE_WIDTH)
from steps.video_camera_motion import VideoMotion


class ImageRowBuilder:
    frames: np.ndarray
    motions: VideoMotion

    def __init__(self, frames, motions, intervals, video_file_path):
        cv2.setNumThreads(N_CPUS)
        self.motions = motions
        self.intervals = intervals
        self.video_file_path = video_file_path
        if LOAD_VIDEO_TO_RAM:
            self.frames = frames
            self.width, self.height = frames.shape[2], frames.shape[1]
        else:
            self.video_capture = cv2.VideoCapture(video_file_path)
            self.width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def construct_rows(self):
        """
        Constructs image rows by processing video frames.
        """
        logging.info(f"Processing RowBuilder for: {self.video_file_path}\n")
        rows = []
        for i, interval in enumerate(self.intervals):
            mn, mx = interval
            start = mn + (mx - mn) // 2 - self.motions.get_frames_per360() // 2
            end = start + self.motions.get_frames_per360()
            file_path = os.path.join(OUTPUT_FOLDER,
                                     os.path.splitext(os.path.basename(self.video_file_path))[0] + f"-oio-{i}.png")
            if not os.path.isfile(file_path):
                row = self.construct_row(int(start), int(end), i)
                rows.append(row)

        if TESTING_MODE:
            for i, row in enumerate(rows):
                file_path = os.path.join(OUTPUT_FOLDER,
                                         os.path.splitext(os.path.basename(self.video_file_path))[0] + f"-oio-{i}-pre_sin.png")
                iio.imwrite(file_path, row.astype(np.uint8))

        rows = self.remove_sin_transform(rows)

        if TESTING_MODE:
            for i, row in enumerate(rows):
                file_path = os.path.join(OUTPUT_FOLDER,
                                         os.path.splitext(os.path.basename(self.video_file_path))[0] + f"-oio-{i}.png")
                iio.imwrite(file_path, row.astype(np.uint8))

        return rows

    def construct_row(self,
                      start: int,
                      end: int,
                      i: int,
                      blended_pixels_per_frame=BLENDED_PIXELS_PER_FRAME,
                      blended_pixels_shift=BLENDED_PIXELS_SHIFT):
        """
        Constructs a single row image from video frames.

        Args:
            start (int): Starting frame index.
            end (int): Ending frame index.
            blended_pixels_per_frame (int): Pixels blended per frame.
            blended_pixels_shift (int): Shift for blending.

        Returns:
            np.ndarray: Constructed row image.
        """
        if end - start <= 0:
            raise IOError("Unsupported input data.")

        frame_size = (self.width, self.height)

        shift_per_frame = self.motions.get_horizontal_speed()
        frames_per_360_deg = self.motions.get_frames_per360()
        direction = self.motions.get_direction()


        image_part = self.height

        if LOAD_VIDEO_TO_RAM:
            getFrame = self.getFrameFromRAM
        else:
            getFrame = self.getFrameFromVidCap

        offset = max(0, math.ceil(start - (blended_pixels_per_frame // 2) / shift_per_frame))
        n_frames = math.ceil(frames_per_360_deg + (blended_pixels_per_frame - 1) / shift_per_frame)

        frame_shift_to_pixels_total = math.ceil(n_frames * shift_per_frame) + (blended_pixels_per_frame - 1) * 2
        row_image = np.zeros(
            (frame_size[0],
             frame_shift_to_pixels_total))

        weight_matrix = np.zeros(row_image.shape)

        for frameNo in tqdm(range(0, n_frames), desc="Building row image"):
            image = getFrame(offset + frameNo)  # shape (h, w), grayscale

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
            movementses = self.calculate_movements(rows)
            movementses = self.detrend_movements(movementses)
            params = self.fitSin(movementses)
            logging.debug(f"\nParameters for sinusoidal transformation:\n{params}\n")
            rows = self.remove_sinusoidal_transformation(rows, params)

        return rows

    def calculate_movements(self, images):
        """
        Calculates cumulative movements between columns from cumulative intensities from stripes from row image.

        Args:
            images (list[np.ndarray]): List of grayscale image frames.

        Returns:
            list[np.ndarray]: List of cumulative movements for each image frame.
        """
        movements_per_image = []
        images = images.copy()
        moving_down = self.motions.is_moving_down()
        for j, image in enumerate(images):
            if moving_down and j == 0:
                image = image[image.shape[0] // 4:, :]
            elif not moving_down and j == len(images) - 1:
                image = image[:-(image.shape[0] // 4), :]

            if VERBOSE and j == 1:
                image_crop = image[:, :image.shape[0]]

                center_x = image.shape[0] // 2 - STRIPE_WIDTH // 2

                fig, ax = plt.subplots(figsize=(10, 8))
                ax.imshow(image_crop, cmap="gray")

                rect1 = patches.Rectangle((center_x-10, 0), STRIPE_WIDTH, image.shape[0], linestyle='--', alpha=0.7,
                                          edgecolor='red', facecolor='none', label='Left Stripe')
                ax.add_patch(rect1)
                rect2 = patches.Rectangle((center_x+10, 0), STRIPE_WIDTH, image.shape[0], linestyle='--', alpha=0.7,
                                          edgecolor='blue', facecolor='none', label='Right Stripe')
                ax.add_patch(rect2)
                ax.set_xlabel("X position")
                ax.set_ylabel("Y position")
                ax.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_FOLDER, "fig_stripe_highlight.png"))
                plt.close()

            image = np.hstack((image,) * IMAGE_REPEATS)

            movements = []

            for i in range(0, image.shape[1] - STRIPE_WIDTH - SINUSOID_SAMPLING, SINUSOID_SAMPLING):
                # Extract vertical stripes
                left_stripe = image[:, i:STRIPE_WIDTH + i]
                right_stripe = image[:, i + SINUSOID_SAMPLING:STRIPE_WIDTH + i + SINUSOID_SAMPLING]

                # Sum pixel intensities along the x-axis
                left_profile = np.sum(left_stripe, axis=1)
                right_profile = np.sum(right_stripe, axis=1)

                left_profile_smoothed = gaussian_filter1d(left_profile, sigma=19)
                right_profile_smoothed = gaussian_filter1d(right_profile, sigma=19)

                distance = 150
                # Detect peaks
                peaks_left, _ = find_peaks(left_profile_smoothed, distance=distance)
                peaks_right, _ = find_peaks(right_profile_smoothed, distance=distance)

                if VERBOSE and j == 1 and i == center_x-10:
                    plt.figure(figsize=(10, 5))
                    plt.plot(left_profile, label="Raw Profile of Left Stripe", color='red', alpha=0.7)
                    plt.plot(right_profile, label="Raw Profile of Right Stripe", color='blue', alpha=0.7)
                    plt.plot(left_profile_smoothed, label="Smoothed Profile of Left Stripe", color='red', alpha=0.7, linestyle="--")
                    plt.plot(right_profile_smoothed, label="Smoothed Profile of Right Stripe", color='blue', alpha=0.7, linestyle="--")
                    plt.plot(peaks_left, left_profile_smoothed[peaks_left], "rx", alpha=0.6, label="Detected Peaks of Left Stripe")
                    plt.plot(peaks_right, right_profile_smoothed[peaks_right], "bx", alpha=0.6, label="Detected Peaks of Right Stripe")
                    plt.xlabel("Y position")
                    plt.ylabel("Sum of Intensities")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(OUTPUT_FOLDER, "fig_peaks_profile.png"))
                    plt.close()

                try:
                    if peaks_left.shape[0] > peaks_right.shape[0]:
                        movement1 = np.mean(peaks_left[1:] - peaks_right)
                        movement2 = np.mean(peaks_left[:-1] - peaks_right)
                        if abs(movement1) > abs(movement2):
                            movement = movement2
                        else:
                            movement = movement1

                    elif peaks_left.shape[0] < peaks_right.shape[0]:
                        movement1 = np.mean(peaks_left - peaks_right[:-1])
                        movement2 = np.mean(peaks_left - peaks_right[1:])
                        if abs(movement1) > abs(movement2):
                            movement = movement2
                        else:
                            movement = movement1
                    else:
                        movement = np.mean(peaks_left - peaks_right)

                    if abs(movement) > 5:
                        movement = movements[-1]
                    movements.append(movement)
                except:
                    movements.append(movements[-1])
                    logging.debug(f"Uncomparable peaks for image {j}, column {i}, {len(peaks_left)} vs {len(peaks_right)}")

            cumulative = np.cumsum(movements)

            if VERBOSE and j == 1:

                plt.figure(figsize=(10, 5))
                plt.plot(cumulative)
                plt.xlabel("Stripe Index")
                plt.ylabel("Cumulative Vertical Shift [px]")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_FOLDER, "fig_cumulative_shift.png"))
                plt.close()

            movements_per_image.append(cumulative)

        return movements_per_image

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

            if VERBOSE:
                # Plot to visualize detrending
                plt.figure(figsize=(10, 5))
                plt.plot(x, movements, label="Original Data")
                plt.plot(x, linear_trend, label="Fitted Line (Trend)")
                plt.plot(x, movements - linear_trend, label="Detrended Data")
                plt.legend()
                plt.xlabel("Stripe Index")
                plt.ylabel("Movement")
                plt.title("Detrending the Cumulative Movement")
                plt.grid(True)
                plt.savefig(
                    os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(self.video_file_path)[0]}_oio_{i}_detrend" + ".png"))
                plt.close()

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
            freq = 2 * IMAGE_REPEATS * np.pi / len(movements)

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

            if VERBOSE:
                y_fit = custom_rotated_sinusoid(x, *params)
                r2 = r2_score(movements, y_fit)
                logging.debug(f"R² score of fit: {r2:.4f}")
                # Compute residuals
                residuals = movements - y_fit
                logging.debug(f"Mean residual: {np.mean(residuals):.4f}")
                logging.debug(f"Std of residuals: {np.std(residuals):.4f}")

                param_errors = np.sqrt(np.diag(pcov))

                for name, err in zip(["A", "C", "D"], param_errors):
                    logging.debug(f"Standard error {name}: {err:.4f}")

                plt.figure()
                plt.plot(x, residuals, label="Residuals")
                plt.axhline(0, color="black", linestyle="--")
                plt.title("Fit Residuals")
                plt.xlabel("X")
                plt.ylabel("Error")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_FOLDER, "fig_residuals.png"))
                plt.close()

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

    def getFrameFromRAM(self, i):
        return self.frames[i]

    def getFrameFromVidCap(self, i):
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, frame = self.video_capture.read()

        if not success or frame is None:
            logging.critical(f"Failed to read frame {i}")
            raise IOError(f"Failed to read frame {i}")

        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
