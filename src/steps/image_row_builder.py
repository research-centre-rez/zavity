import logging
import math
import os
import cv2
import imageio.v3 as iio
import numpy as np
from scipy.optimize import brute
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from tqdm.auto import tqdm

from config.config import (BLENDED_PIXELS_PER_FRAME, BLENDED_PIXELS_SHIFT, OUTPUT_FOLDER, TESTING_MODE, SINUSOID_SAMPLING)
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
        for iid, interval in enumerate(self.intervals):
            int_start, int_end = interval
            start = int_start + (int_end - int_start) // 2 - self.motions.get_frames_per360() // 2
            end = start + self.motions.get_frames_per360()
            file_path = os.path.join(OUTPUT_FOLDER,
                                     os.path.splitext(os.path.basename(self.video_file_path))[0] + f"-oio-{iid}.png")
            if not os.path.isfile(file_path):
                row, frame_map = self.construct_row(int(start), int(end), iid, return_frame_map=True)
                yshifts_compensated = ImageRowBuilder.compute_column_shifts_dic(row)
                row_compensated = ImageRowBuilder.remove_column_shifts_dic(row, yshifts_compensated)
                iio.imwrite(file_path, row_compensated.astype(np.uint8))
                file_path = os.path.join(OUTPUT_FOLDER,
                                         os.path.splitext(os.path.basename(self.video_file_path))[
                                             0] + f"-oio-{iid}-frame-map.csv")
                with open(file_path, "w") as f:
                    for frame_no in frame_map:
                        f.write(f"{frame_no + int_start}\n")
                rows.append(row_compensated)
            else:
                row = iio.imread(file_path)
                rows.append(row)

        return rows

    def construct_row(self,
                      start: int,
                      end: int,
                      row_id: int,
                      blended_pixels_per_frame=BLENDED_PIXELS_PER_FRAME,
                      blended_pixels_shift=BLENDED_PIXELS_SHIFT,
                      return_frame_map=False):
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
        n_frames = math.ceil(frames_per_360_deg + 2 * (blended_pixels_per_frame - 1) / shift_per_frame)

        frame_shift_to_pixels_total = math.ceil(n_frames * shift_per_frame) + (blended_pixels_per_frame - 1) * 2
        row_image = np.zeros((frame_size[0], frame_shift_to_pixels_total))
        frame_map = np.zeros((frame_shift_to_pixels_total,))

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
                frame_map[crop_x_start:crop_x_end] = frameNo
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

        valid_data = np.where(np.sum(weight_matrix, axis=0) != 0)[0]
        start_valid_col = np.argmin(valid_data)
        end_valid_col = np.argmax(valid_data)

        start_col = np.max([(end_valid_col - row_size) // 2, start_valid_col])
        end_col = np.min([start_col + row_size, end_valid_col])
        row_image = (row_image / weight_matrix)[:, start_col:end_col]

        if return_frame_map:
            return np.copy(row_image), frame_map[start_col:end_col]
        else:
            return np.copy(row_image)

    @staticmethod
    def compute_column_shifts_dic(image):
        # TODO: move constants somewhere
        # Row image is blurred first (before column registration), blur along the vertical should be higher than horizontal
        # This bluring possitively affects convergence in case of gradient method is used
        # NOTE: Finally, brute method is used, maybe this is not necessary (but it eliminates noisiness of the signal as well)
        BLUR_VERTICAL_KERNEL = 101
        BLUR_HORIZONTAL_KERNEL = 51
        BLUR_SIGMA = 121
        # It is not clear what is a good smoothing factor. Large numbers have a problem with the beginning and the end of a signal
        # Median filter is used for smoothing y-shifts computed for each column
        MEDIAN_FILTER_SIZE = 1001
        # Registered are only sparse columns of the image.
        REGISTRATION_REFERENCE_FRAME_WIDTH = 1000
        REGISTRATION_COLUMN_SPARSITY = 100
        REGISTRATION_COLUMN_OFFSET = 10
        # registration is done only on horizontal stripe of the image (to prevent work with the image boundary)
        REGISTERED_CROP = [600, 1200]

        # Computation of yshift for each column runs in three steps:
        # 1) columns are registered with respect of registered columns to the left
        # 2) because first REGISTRATION_REFERENCE_FRAME has no registered predecessors these columns are registered in
        # second run backwards (so respected registered predecessors are to the right)
        # 3) because the threded socket is circular the first and last column horizontal position must match.
        # This is estimated in the third step.
        #
        # When raw y-shifts are computed, linear approximation is created and columns are shifted along this "thread
        # axis". Smoothing is applied to reduce effect of outliers.

        I = cv2.GaussianBlur(np.nan_to_num(image, copy=True, nan=0).astype(np.uint8), (BLUR_VERTICAL_KERNEL, BLUR_HORIZONTAL_KERNEL), BLUR_SIGMA)
        yshifts = [0]
        yshifts_cum = [0]
        for column in tqdm(range(1, I.shape[1]), desc="Removing column shift (forward run)"):
            def yshift(y):
                sum = 0
                for en, yshifts_len in enumerate(np.arange(REGISTRATION_COLUMN_OFFSET, REGISTRATION_REFERENCE_FRAME_WIDTH, REGISTRATION_COLUMN_SPARSITY)):
                    if len(yshifts) > yshifts_len:
                        relative_shift = int(yshifts_cum[-1] - yshifts_cum[-yshifts_len] + y[0])
                        sampled = I[REGISTERED_CROP[0] + relative_shift: REGISTERED_CROP[1] + relative_shift, column]
                        sum += np.sum(np.abs(np.diff(sampled) - np.diff(
                            I[REGISTERED_CROP[0]: REGISTERED_CROP[1], column - yshifts_len])))
                    else:
                        break
                return sum

            dic = brute(yshift, ranges=[slice(-1, 2, 1)])
            yshifts.append(dic[0])
            yshifts_cum.append(yshifts_cum[-1] + dic[0])

        # Above code works well for columns 1000+. To fix the left part of the image we run the same code backwards for these columns
        for column in tqdm(np.arange(REGISTRATION_REFERENCE_FRAME_WIDTH, -1, -1),
                           desc="Removing column shift (backward run)"):
            def yshift(y):
                sum = 0
                for en, yshifts_len in enumerate(
                        np.arange(column + REGISTRATION_COLUMN_OFFSET, column + REGISTRATION_REFERENCE_FRAME_WIDTH, REGISTRATION_COLUMN_SPARSITY)):
                    relative_shift = int(yshifts_cum[column + 1] - yshifts_cum[yshifts_len] + y[0])
                    sampled = I[REGISTERED_CROP[0] + relative_shift: REGISTERED_CROP[1] + relative_shift, column]
                    sum += np.sum(
                        np.abs(np.diff(sampled) - np.diff(I[REGISTERED_CROP[0]: REGISTERED_CROP[1], yshifts_len])))
                return sum

            dic = brute(yshift, ranges=[slice(-1, 2, 1)])
            yshifts[column] = -dic[0]
            yshifts_cum[column] = yshifts_cum[column + 1] + dic[0]

        # Cyclic overlap (end to start)
        def yshift(y):
            """This method was finetuned for many videos. It generates reasonable global minima for correct row shift"""
            y_shift = int(y[0])
            L = I[REGISTERED_CROP[0] + y_shift: REGISTERED_CROP[1] + y_shift, :40]
            R = I[REGISTERED_CROP[0]: REGISTERED_CROP[1], -40:]
            return np.sum(np.abs(
                L / (R + 1) - 1
            ))

        cyclic_overlap = brute(yshift, ranges=[slice(-200, 200, 1)])[0]

        logging.debug(f"Cyclic overlap: {cyclic_overlap}")

        yshifts_smooth = median_filter(np.cumsum(yshifts), MEDIAN_FILTER_SIZE,
                                       mode="nearest")
        total_shift = cyclic_overlap - (yshifts_smooth[20] - yshifts_smooth[-20])
        thread_slope = total_shift / (I.shape[1] + 1)

        y_diffs = [x * thread_slope - yshifts_smooth[x] for x in range(I.shape[1])]
        b = -(np.max(y_diffs) + np.min(y_diffs)) / 2
        yshifts_compensated = yshifts_smooth - np.polyval([thread_slope, b], np.arange(I.shape[1]))

        return yshifts_compensated

    @staticmethod
    def remove_column_shifts_dic(image, yshifts):
        row = np.zeros_like(image)
        crop = np.max(np.abs(-yshifts.astype(int)))
        print(f"Cropping row image by {crop} pixels (height {row.shape[0]})")

        for column in tqdm(np.arange(image.shape[1]), desc="Building row image"):
            row[:, column] = np.roll(image[:, column], -yshifts[column].astype(int))
        if crop == 0:
            return row
        return row[crop:-crop]

    @staticmethod
    def remove_column_shifts(image, shifts):
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
