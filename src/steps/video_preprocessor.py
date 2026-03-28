import os
import logging
import subprocess
import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import LiteralString
from scipy.signal import find_peaks
from tqdm.auto import tqdm
import multiprocessing

from config.config import (PREPROCESSOR_SAMPLING, OUTPUT_FOLDER, RECTIFICATION_PARAMS_FOLDER, RECTIFY,
                           PREPROCESSOR_DOWNSCALE, SEGMENT_TYPE_TH, REMOVE_ROTATION, CODEC, EXT, VERBOSE)
from scripts.main import timing
from signal_utils import find_step_edge_subsample, circular_signal_derivative, find_slope_change_broken_line, \
    circular_signal
from steps.adaptive_frame_cropping import AdaptiveFrameCropper, CROPPED_FRAME_SIDE_PX
from steps.video_rectifier import _load_calibration_parameters
import pandas as pd


class VideoPreprocessor:
    """
    A class for preprocessing videos by analysing frames:
    - estimating center of the scene
    - computing threads orientation in the scene
    - finding frame numbers where is a change in engine mode (shift vs. rotation)
    - splitting the video accordingly
    - applying rotation corrections

    Outputs a processed video and computed centers/orientations.

    Attributes:
        video_name (str): Name of the input video file.
        frames_crop_centers (np.ndarray): array containing coordinates of a frame center (for cropping valid data)
        output_video_file_path (str): Path to the output processed video file.
        angles (np.ndarray): List of computed angles for each frame.
        borderBreakpoints (list): List of border breakpoints derived from the video.
        breakpoints (np.ndarray): Detected breakpoints in the video based on angles.
        segment_type (np.ndarray): Segment type array representing trends in angles.
        rotation_sequences (list): List of rotation sequences derived from the video.
    """
    video_name: str
    frames_crop_centers: np.ndarray
    output_video_file_path: LiteralString | str | bytes
    angles: np.ndarray
    flips: np.ndarray
    borderBreakpoints: list
    breakpoints: np.ndarray
    segment_type: np.ndarray
    rotation_sequences: list

    def __init__(self, video_path, frames_crop_centers):
        """
        Initializes the VideoPreprocessor.

        Args:
            video_path (str): Path to the input video file.
        """
        self.video_path = video_path
        self.frames_crop_centers = frames_crop_centers
        self.angles = None
        self.flips = None
        self.clean_angles = None
        self.video_name = os.path.basename(self.video_path)
        self.output_video_file_path = self._dump_path('preprocessed', EXT)
        self.borderBreakpoints = []

        video_capture = cv2.VideoCapture(self.video_path)
        self.num_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_capture.release()

        _, mtx, newcameramtx, distortion, _, _ = _load_calibration_parameters(RECTIFICATION_PARAMS_FOLDER)
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(mtx, distortion, None, newcameramtx,
                                                           (self.video_width, self.video_height), 5)

    def process_or_load(self):
        """
        Starts processing the video.
        If video already exists, it loads computed data which is needed in next steps.
        """
        if REMOVE_ROTATION:
            logging.info(f"Pre-processing video: {self.video_name}")
            self._get_orientation_and_breakpoints()
            self.parallel_rotation_compensation()
        else:
            self.output_video_file_path = self.video_name


    def _get_orientation_and_breakpoints(self):
        """
        Loads or computes rotation per frames (if required), threads orientation, and breakpoints for the video.
        """
        if os.path.isfile(self._dump_path("full_angles", extension="csv")):
            logging.info("Angles already estimated. Loading data ...")
            angles = self.load_csv("full_angles")
            self.angles = angles[:, :2]
            self.clean_angles = angles[:, 2]
        else:
            logging.info("Angles have to be estimated. Running estimation process ...")
            with timing("Estimate angles"):
                self.angles = self.parallel_threads_orientation_est(step=1, angle_precision=1800, hough_treshold=200)
                self.dump_csv("full_angles", pd.DataFrame(self.angles, columns=["frame number", "angle (rad)"]))

        if (os.path.isfile(self._dump_path("breakpoints", extension="csv"))):
            self.breakpoints = self.load_csv("breakpoints").astype(int)
            self.rotation_sequences = [[seq_start, seq_end] for seq_start, seq_end, seq_type in self.breakpoints if seq_type == 1]
            logging.debug(f"Loaded {self._dump_path('breakpoints')}: {self.breakpoints}")
        else:
            with timing("Compute Breakpoints"):
                rotation_sequences, shift_sequences, clean_angles = self.split_video_according_engine_movement_type()
                self.clean_angles = clean_angles
                self.dump_csv("full_angles",
                              pd.DataFrame(np.stack([self.angles[:, 0], self.angles[:, 1], clean_angles], axis=1),
                                           columns=["frame number", "angle (deg)", "clean angle (deg)"]))
                self.breakpoints = np.concatenate([
                    np.column_stack([np.array(rotation_sequences), np.ones((len(rotation_sequences),))]),
                    np.column_stack([np.array(shift_sequences), np.zeros((len(shift_sequences),))])
                ])
                self.breakpoints = self.breakpoints[self.breakpoints[:, 0].argsort()].astype(int)
                self.dump_csv("breakpoints", pd.DataFrame(self.breakpoints, columns=["sequence start", "sequence end",
                                                                                     "sequence type (rot 1, shift 0)"]))

        self.plot_angles()

    @staticmethod
    def circular_median_180(angles_deg):
        angles_rad = np.deg2rad(angles_deg)

        # double-angle mapping
        angles2 = 2 * angles_rad

        def angular_distance(a, b):
            d = np.angle(np.exp(1j * (a - b)))
            return np.abs(d)

        # brute-force over candidates
        costs = []
        for a in angles2:
            cost = np.sum(angular_distance(angles2, a))
            costs.append(cost)

        best = angles2[np.argmin(costs)]

        return np.rad2deg(best / 2)

    @staticmethod
    def sequential_thread_orientation_est(params):
        """
        Analyzes the orientation of frames in a video using edge detection and Hough Line Transform. Expects that
        frame centers are already computed.

        This method processes a specified range of video frames to detect and calculate the dominant
        orientation of edges in each frame. The process involves adaptive cropping, resizing, Gaussian
        blurring, edge detection using the Canny algorithm, and applying the Hough Line Transform to
        extract line orientations. The resulting angles for each frame are returned as a list.

        Parameters:
            params (tuple): A tuple containing the following elements:
                - video_path (str): Path to the video file.
                - start_frame (int): The starting frame number for processing.
                - end_frame (int): The ending frame number for processing.
                - angle_precision (float): The precision for angle calculations in the Hough Transform.
                - hough_threshold (int): The threshold for the Hough Transform to detect lines.
                - apply_abs (bool): Whether to apply absolute value to the computed angle.
                - frames_center (list[tuple[int, int, int]]): List of tuples, each containing a
                  frame number and the center coordinates (cx, cy) for cropping.

        Returns:
            list[tuple[int, Union[float, None]]]: A list of tuples, where each tuple contains:
                - frame_no (int): The frame number.
                - angle_median (Union[float, None]): The dominant orientation angle in degrees
                  for the frame or None if no lines were detected.

        Raises:
            AssertionError: If frame numbers from the frames_center input do not match the frame
            numbers being processed.
        """
        video_path, start_frame, end_frame, angle_precision, hough_threshold, apply_abs, frames_center = params
        angles = []
        cap = cv2.VideoCapture(video_path)  # it is necessary to instantiate video capture for each thread separately
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for frame_no in tqdm(range(start_frame, end_frame), total=end_frame - start_frame, desc=f"Frames from {start_frame} to {end_frame}"):
            success, full_frame = cap.read()
            if not success:
                cap.release()
                break

            fno, cx, cy = frames_center[frame_no - start_frame]
            assert fno == frame_no, "Error in frame numbers"
            frame = AdaptiveFrameCropper.crop(full_frame, cx, cy)

            frame = cv2.resize(frame, (
                frame.shape[1] // PREPROCESSOR_DOWNSCALE,
                frame.shape[0] // PREPROCESSOR_DOWNSCALE))
            frame = cv2.GaussianBlur(frame, (5, 5), 1)

            # Edge detection
            edges = cv2.Canny(frame, 70, 180)

            # Hough Transform
            lines = cv2.HoughLinesP(edges, 1, np.pi / angle_precision, hough_threshold,
                                    minLineLength=frame.shape[0] / 2.5,
                                    maxLineGap=frame.shape[0] / 3)
            if lines is None:
                angles.append((frame_no, None))
                continue

            # Extract angles and compute dominant
            raw_angles = []

            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle_rad = np.arctan2(y2 - y1, x2 - x1)
                raw_angles.append(np.rad2deg(angle_rad) )

            # Classic median is extremely unstable for angles around 90 (-90)
            angle_median = VideoPreprocessor.circular_median_180(raw_angles)
            if apply_abs:
                angle_median = np.abs(angle_median)
            angles.append((frame_no, angle_median))
        return angles

    def _nan_helper(self, y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    def parallel_threads_orientation_est(self, start=0, end=None, step=PREPROCESSOR_SAMPLING, angle_precision=90, hough_treshold=80,
                                         apply_abs=False):
        """
        Computes the orientation of threads in parallel using multiprocessing.

        This method divides the computation of thread orientation across multiple CPU cores by leveraging
        multiprocessing. It segments the frames of the video into manageable chunks and processes
        them concurrently to estimate the orientations efficiently. The computation uses a Hough Transform
        based approach to analyze frames and identify angles with specified precision and thresholds.

        Parameters:
            start (int, optional): The starting frame index for computations. Defaults to 0.
            end (int, optional): The ending frame index for computations. If None, the method
                uses the total number of frames available in the video.
            step (int, optional): The step size to use when iterating through frames.
                Defaults to PREPROCESSOR_SAMPLING.
            angle_precision (int, optional): The precision of angle estimations. Defaults to 90.
            hough_treshold (int, optional): The threshold parameter for the Hough transform.
                Defaults to 80.
            apply_abs (bool, optional): Flag indicating whether to take the absolute value with respect
                to computed orientations. Defaults to False.

        Returns:
            numpy.ndarray: An array of tuples (frame_index, angle), where each tuple contains the frame
                number and the estimated angle corresponding to that frame.
        """
        if end is None:
            end = self.num_frames

        cpus = multiprocessing.cpu_count() - 1
        with multiprocessing.Pool(cpus) as pool:
            results = list(tqdm(pool.imap(self.sequential_thread_orientation_est,
                                          [(self.video_path, frame_no, frame_no + (end - start) // cpus, angle_precision,
                                            hough_treshold // PREPROCESSOR_DOWNSCALE, apply_abs,
                                            self.frames_crop_centers[frame_no: frame_no + (end - start) // cpus])
                                           for frame_no in range(start, end, (end-start) // cpus)]),
                                total=cpus, #(end-start)/step,
                                desc=f"Computing angles from {start} to {end} with step {step}")
                           )

        computed_angles = np.array(sorted([(int(frame_no), float(angle) if angle is not None else np.nan)
                                           for records in results for frame_no, angle in records], key=lambda x: x[0]))
        # interpolate missing values
        nans, x = self._nan_helper(computed_angles[:,1])
        computed_angles[nans, 1] = np.interp(x(nans), x(~nans), computed_angles[~nans, 1])

        logging.debug(f"Angles calculated from {start} to {end} with step {step}\n")
        return computed_angles

    def split_video_according_engine_movement_type(self, threshold=SEGMENT_TYPE_TH):
        """
        Refines rotation angles for each frame to be smooth.
        Find-out frames where engine mode (shift vs. rotation) is changing.

        return:
            np.ndarray, np.ndarray, nd.array:
            - Array of rotation sequences,
            - array of shift sequences,
            - array of cleaned continous angles.
        """
        MIN_FRAMES_PER_SEQUENCE = 200  # i.e. 16 seconds

        # Find segments of increasing/decreasing angles.
        # Because the change is noisy, we use PREPROCESSOR_SAMPLING to accumulate the change.
        # However, PREPROCESSOR_SAMPLING affects the precision of the change detection,
        # which has to be fine-tuned (see below).
        angle_derivative = circular_signal_derivative(self.angles[:, 1], PREPROCESSOR_SAMPLING, 180)

        segments_type = np.zeros_like(angle_derivative)
        segments_type[:] = np.NaN
        segments_type[angle_derivative > threshold] = 1  # Increasing
        segments_type[angle_derivative < -threshold] = -1  # Decreasing
        segments_type[np.abs(angle_derivative) < threshold / 2] = 0

        # Estimate breakpoints and establish sequences
        shift_sequences = []  # video intervals where the tubus with the mirror is not rotating
        seq_zero_len = 0
        for frame_no, segment_type in enumerate(segments_type):
            if np.isnan(segment_type):  # segment type is not specified, change is present, but not enough
                continue
            if segment_type == 0:  # the angle is not changing between two consecutive frames
                if seq_zero_len == 0:
                    shift_start = frame_no
                seq_zero_len += 1
            elif seq_zero_len != 0:
                if seq_zero_len > MIN_FRAMES_PER_SEQUENCE: # more than 100 frames where rotation is not present
                    MAX_ERROR = 1.1
                    correct_angles = False
                    # NOTE: Here is necessary to compensate the sampling
                    # - start of the rotation can be somewhere between two sampling points
                    # - idea is to find change of slope in raw angle sequence around already found point
                    start_lower_bound = np.max([0, shift_start - PREPROCESSOR_SAMPLING])
                    start_upper_bound = np.min([len(self.angles), shift_start + PREPROCESSOR_SAMPLING])
                    raw_signal = self.angles[start_lower_bound: start_upper_bound, 1]
                    if np.mean(np.abs(raw_signal)) > MAX_ERROR * np.mean(raw_signal):
                        motion_start = find_slope_change_broken_line(circular_signal(raw_signal))
                        correct_angles = True
                    else:
                        motion_start = find_slope_change_broken_line(raw_signal)

                    end_lower_bound = np.max([0, frame_no - PREPROCESSOR_SAMPLING])
                    end_upper_bound = np.min([len(self.angles), frame_no + PREPROCESSOR_SAMPLING])
                    raw_signal = self.angles[end_lower_bound: end_upper_bound, 1]

                    if np.mean(np.abs(raw_signal)) > MAX_ERROR * np.mean(raw_signal):
                        motion_end = find_slope_change_broken_line(circular_signal(raw_signal))
                        correct_angles = True
                    else:
                        motion_end = find_slope_change_broken_line(raw_signal)

                    motion_start_abs = np.round(motion_start["x0"]).astype(int) + start_lower_bound
                    motion_end_abs = np.round(motion_end["x0"]).astype(int) + end_lower_bound
                    logging.debug(f"Increasing precision of shift breakpoints: {shift_start} -> {motion_start_abs}, {frame_no} -> {motion_end_abs}")
                    shift_sequences.append((motion_start_abs, motion_end_abs))
                    if correct_angles:
                        self.angles[motion_start_abs: motion_end_abs, 1] = circular_signal(self.angles[motion_start_abs: motion_end_abs, 1], shift=90)
                seq_zero_len = 0

        if VERBOSE:
            plt.figure(figsize=(15, 5))
            plt.plot(self.angles[:, 1], alpha=0.5)
            for seq in shift_sequences:
                plt.axvline(seq[0], color="red")
                plt.axvline(seq[1], color="green")
            plt.show()

        # TODO: move this cleanup into separate methods
        clean_angles = np.copy(self.angles[:,1])
        # Rotation sequences have a similar length:
        rot_sequences_length = np.median([n[0] - p[1]
                                          for p, n in zip(shift_sequences[:-1], shift_sequences[1:])]).astype(int)

        # Build the inverse => rotation sequences
        rotation_sequences = []
        # NOTE:
        # sequence safety padding is good for debug, but not so good for real usage,
        # it generates new problems. The padded part should be video, where nothing is moving otherwise:
        # - motion of frames jumps (no smooth changes)
        # - rotation is missing (it is cliped)
        # Finally, this was solved by better detection of shift/rotation in the angles sequence.
        SEQENCE_SAFETY_PADDING = 0  # This should be dropped once the code is stable.
        # Before the first and after the last shift sequences could be rotation sequence.
        # Heuristic below adds a rotation sequence before and after shift sequences when there are enough frames present.
        # If results are not good - crop the video or improve this section
        start = shift_sequences[0][0] - rot_sequences_length
        for seq in shift_sequences:
            if start < 0:
                start = seq[1]
                continue
            clean_angles[seq[0]: seq[1]] = np.median(self.angles[seq[0] + SEQENCE_SAFETY_PADDING: seq[1] - SEQENCE_SAFETY_PADDING, 1])
            assert start < seq[0] - SEQENCE_SAFETY_PADDING
            rotation_sequences.append((start + SEQENCE_SAFETY_PADDING, seq[0] - SEQENCE_SAFETY_PADDING))
            start = seq[1]
        if np.nansum(np.abs(segments_type[start:])) > MIN_FRAMES_PER_SEQUENCE:
            rotation_sequences.append((start + SEQENCE_SAFETY_PADDING, start + rot_sequences_length - SEQENCE_SAFETY_PADDING))

        # Measured angles in rotation sequences contains discontinuities. This is handled here.
        flips = 0  # points of HoughLines discontinuity (-90° = 90°)
        for start, end in rotation_sequences:
            data = self.angles[start:end, 1]
            extremas = find_peaks(data, distance=200, height=87)[0]

            beginning = 0
            noisy_data = np.zeros((end - start, ))
            for e in extremas:
                noisy_data[beginning: e] = 180 * flips + data[beginning: e]
                beginning = e
                flips += 1
            noisy_data[beginning:] = 180 * flips + data[beginning:]

            # Replace noisy signal with a polynomial fit
            data_x = np.arange(end - start)
            approximation = np.polyfit(data_x, noisy_data, 4)
            clean_angles[start: end] = np.polyval(approximation, np.arange(end - start))

            if flips % 2 == 1:
                clean_angles[end:] += 180

        self.rotation_sequences = rotation_sequences

        # The idea here is that measured angles form continuous (noisy) sequence. But HoughLines produces angles between
        # -90, 90 (i.e. there is a discontinuity). Here we create the continuous sequence by adding k * 180° in every
        # discontinuous point to stitch the signal.
        ca_continuous = np.copy(clean_angles)
        ca_continuous[:rotation_sequences[0][0]] = ca_continuous[rotation_sequences[0][0]]
        for prev, next in zip(rotation_sequences[:-1], rotation_sequences[1:]):
            diff = (clean_angles[prev[1] - 1] - clean_angles[next[0]])
            diff_rounded = np.round(diff / 180)
            if diff_rounded != 0:
                ca_continuous[next[0]:] += diff_rounded * 180
            ca_continuous[prev[1]:next[0]] = np.mean([ca_continuous[prev[1] - 1], ca_continuous[next[0]]])
        ca_continuous[rotation_sequences[-1][1]:] = ca_continuous[rotation_sequences[-1][1] - 1]

        return rotation_sequences, shift_sequences, ca_continuous

    def get_intervals(self):
        return self.rotation_sequences

    @staticmethod
    def rotation_compensation(params):
        """
        Rotates frames of a video incrementally by applying compensation for spherical distortion
        (if specified) and saves the processed frames to an output video file. The rotation angle
        per frame is controlled by a specified parameter, allowing a smooth transition from one
        angle to another over the given frame range.

        Parameters:
            video_in_path (str): Path to the input video file.
            video_out_path (str): Path to the output video file.
            start (int): The starting frame index of the range to process.
            end (int): The ending frame index of the range to process.
            frame_angle_deg (float): Rotation which should be applied on a frame (deg)
            frames_center (float): The center of each frame
        """
        video_in_path, video_out_path, start, end, frame_angle_deg, frames_center = params

        cap_in = cv2.VideoCapture(video_in_path)
        video_out = cv2.VideoWriter(
            video_out_path,
            apiPreference=cv2.CAP_FFMPEG,
            fourcc=cv2.VideoWriter_fourcc(*CODEC),
            fps=cap_in.get(cv2.CAP_PROP_FPS),
            frameSize=(CROPPED_FRAME_SIDE_PX, CROPPED_FRAME_SIDE_PX),
            params=[
                cv2.VIDEOWRITER_PROP_DEPTH,
                cv2.CV_8U,
                cv2.VIDEOWRITER_PROP_IS_COLOR,
                0,
            ])
        if RECTIFY:
            frame_size_cv2 = (int(cap_in.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_in.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            _, mtx, newcameramtx, distortion, _, _ = _load_calibration_parameters(RECTIFICATION_PARAMS_FOLDER)
            mapx, mapy = cv2.initUndistortRectifyMap(mtx, distortion, None, newcameramtx, frame_size_cv2, 5)

        cap_in.set(cv2.CAP_PROP_POS_FRAMES, start)

        for i in tqdm(range(start, end), desc=f"Rotation compensation {start}-{end}", total=end - start):
            frame_no, cx, cy = frames_center[i - start]
            success, frame = cap_in.read()
            if not success:
                logging.warning(f"Sequence {start} to {end} ends sooner {i}.")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # compensate spherical distortion
            if RECTIFY:
                frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR).astype(np.uint8)
            # compute rotation matrix
            rotate_matrix = cv2.getRotationMatrix2D((cx, cy), frame_angle_deg[i - start], 1)

            rotated_image = AdaptiveFrameCropper.crop(cv2.warpAffine(
                src=frame,
                M=rotate_matrix,
                dsize=frame.shape[::-1],
                flags=cv2.INTER_CUBIC
            ), cx, cy)

            video_out.write(rotated_image.astype(np.uint8))

        video_out.release()
        cap_in.release()
        return video_out_path

    def parallel_rotation_compensation(self):
        if os.path.isfile(self.output_video_file_path):
            logging.info(f"Compensated video already exists: {self.output_video_file_path}")
            return

        run_params = []
        logging.info("Preparing parameters for parallel compensation of rotation")
        prev_end = 0
        for sequence_start, sequence_end, sequence_type in self.breakpoints:
            if sequence_start > prev_end:
                run_params.append((
                    self.video_path,
                    self._dump_path(f"compensated-{prev_end:06d}", extension=EXT),
                    prev_end, sequence_start,
                    self.clean_angles[prev_end: sequence_start],
                    self.frames_crop_centers[prev_end: sequence_start]
                ))
            run_params.append((
                self.video_path,
                self._dump_path(f"compensated-{sequence_start:06d}", extension=EXT),
                sequence_start, sequence_end,
                self.clean_angles[sequence_start: sequence_end],
                self.frames_crop_centers[sequence_start: sequence_end]
            ))
            prev_end = sequence_end

        # each sequence is processed separately
        cpus = multiprocessing.cpu_count() - 1
        with multiprocessing.Pool(cpus) as pool:
            video_parts_paths = list(tqdm(pool.imap(VideoPreprocessor.rotation_compensation, run_params)))

        cv2.destroyAllWindows()

        with open(self._dump_path("video_parts", extension="txt"), "wt") as ffmpeg_concat_instructions:
            for part_path in sorted(video_parts_paths):
                ffmpeg_concat_instructions.write(f"file '{part_path}'\n")

        logging.info("Joining videos with compensated rotation...")
        subprocess.Popen([
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", self._dump_path("video_parts", "txt"),
            "-c", "copy",
            self.output_video_file_path
        ]).wait()

    def plot_angles(self):
        """
        Plots thread's orientations and optionally marks breakpoints and breakpoint candidates.

        Args:
            cleaned_angles (np.ndarray): List of thread's orientations to plot.
            breakpoint_candidates (np.ndarray, optional): List of breakpoint candidates to mark on the plot.
            breakpoints (list, optional): List of breakpoints to mark on the plot.
        """
        plt.figure(figsize=(15, 3))
        plt.plot(self.angles[:, 1], alpha=0.5, label="raw angles")
        plt.plot(self.clean_angles % 90, label="cleaned movement")
        if self.breakpoints is not None:
            for bs, be, bt in self.breakpoints:
                plt.axvline(bs, color="green" if bt == 1 else "red")
                plt.axvline(bt, color="blue" if bt == 1 else "orange")
        plt.title("Video split according to engine movement")
        plt.legend()
        plt.savefig(self._dump_path("angles", "png"))
        plt.show()
        plt.close()

    def get_output_video_file_path(self):
        """
        Retrieves the path for the output processed video file.

        Returns:
            str: Path to the output video file.
        """
        return self.output_video_file_path

    def _dump_path(self, object_name, extension='npy'):
        """
        Generates a path for saving or loading a specific object related to the video.

        Args:
            object_name (str): Name of the object to save/load.
            extension (str): File extension for the object. Defaults to 'npy'.

        Returns:
            str: Path to the file.
        """
        return os.path.join(OUTPUT_FOLDER, os.path.splitext(self.video_name)[0] + f'-{object_name}.{extension}')

    def dump_csv(self, name: str, dataFrame):
        dataFrame.to_csv(self._dump_path(name, extension="csv"), index=False)

    def load_csv(self, name: str):
        array2D = pd.read_csv(self._dump_path(name, extension="csv")).to_numpy()
        return array2D.reshape(-1) if array2D.shape[-1] == 1 else array2D

