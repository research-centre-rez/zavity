import os
import logging
import subprocess
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import LiteralString
from scipy import stats
from scipy.signal import savgol_filter, savgol_coeffs
from tqdm.auto import tqdm

from config.config import (N_CPUS, Y1, Y2, X1, X2, PREPROCESSOR_SAMPLING, ROT_PER_FRAME, PADDING, OUTPUT_FOLDER,
                           RECTIFICATION_PARAMS_FOLDER, RECTIFY, INTERVAL_FILTER_TH, PREPROCESSOR_DOWNSCALE,
                           SEGMENT_TYPE_TH, REMOVE_ROTATION, LOAD_VIDEO_TO_RAM, CODEC, EXT, PITCH_ANGLE,
                           VERBOSE)
from scripts.main import timing
from steps.video_rectifier import _load_calibration_parameters


class VideoPreprocessor:
    """
    A class for preprocessing videos by analysing frames, computing threads orientation per frame, and
    applying rotation corrections. Outputs a processed video and related computed data.

    Attributes:
        video_name (str): Name of the input video file.
        output_video_file_path (str): Path to the output processed video file.
        video_capture (cv2.VideoCapture): Video capture object for reading frames.
        angles (np.ndarray): List of computed angles for each frame.
        borderBreakpoints (list): List of border breakpoints derived from the video.
        calc_rot_per_frame (bool): Flag indicating whether to calculate rotation per frame.
        rotation_per_frame (float): Calculated rotation per frame.
        breakpoints (np.ndarray): Detected breakpoints in the video based on angles.
        segment_type (np.ndarray): Segment type array representing trends in angles.
        frames (np.ndarray): List of frames captured from the video used when processing on RAM.
        processed_frames (np.ndarray): List of processed frames to return from this class used when processing on RAM.
        video_writer (cv2.VideoWriter): Video writer object for writing the processed video when RAM is not big enough.
    """
    video_name: str
    output_video_file_path: LiteralString | str | bytes
    video_capture: cv2.VideoCapture
    angles: np.ndarray
    borderBreakpoints: list
    calc_rot_per_frame: bool
    rotation_per_frame: float
    breakpoints: np.ndarray
    segment_type: np.ndarray
    frames: np.ndarray
    processed_frames: np.ndarray
    video_writer: cv2.VideoWriter

    def __init__(self, video_path, calc_rot_per_frame):
        """
        Initializes the VideoPreprocessor.

        Args:
            video_path (str): Path to the input video file.
            calc_rot_per_frame (bool): Whether to calculate rotation per frame.
        """
        cv2.setNumThreads(N_CPUS)
        self.video_path = video_path
        self.calc_rot_per_frame = calc_rot_per_frame
        self.video_capture = cv2.VideoCapture(self.video_path)
        self.video_name = os.path.basename(self.video_path)
        self.output_video_file_path = os.path.join(OUTPUT_FOLDER,
                                                   os.path.splitext(self.video_name)[0] + '_preprocessed' + EXT)
        self.borderBreakpoints = []
        self.num_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frames = np.empty((self.num_frames, self.video_height, self.video_width), dtype=np.uint8)
        self.processed_frames = np.empty((self.num_frames, Y2 - Y1, X2 - X1), dtype=np.uint8)
        _, mtx, newcameramtx, distortion, _, _ = _load_calibration_parameters(RECTIFICATION_PARAMS_FOLDER)
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(mtx, distortion, None, newcameramtx,
                                                           (self.video_width, self.video_height), 5)

    def process(self):
        """
        Starts processing the video.
        If video already exists, it loads computed data which are needed in next steps.
        """
        if REMOVE_ROTATION:
            if os.path.isfile(self.get_output_video_file_path()) and os.path.isfile(
                    self._dump_path("borderBreakpoints")):
                self.borderBreakpoints = np.load(self._dump_path("borderBreakpoints"))
                logging.debug(f"Loaded {self._dump_path('borderBreakpoints')}\n"
                              f"{self.borderBreakpoints}\n")
            else:
                logging.info(f"Pre-processing video: {self.video_name}")
                if LOAD_VIDEO_TO_RAM:
                    with timing("Load Frames"):
                        self.loadFrames()
                self.load_or_compute()
                self.preprocess_video()
        else:
            self.output_video_file_path = self.video_name

    def loadFrames(self):
        """
        Loads frames to memory for faster processing.
        """
        w = self.video_width  # or X2 - X1 if cropping
        h = self.video_height  # or Y2 - Y1
        frame_size = w * h  # bytes per grayscale frame
        command = [
            "ffmpeg",
            "-y",
            "-threads", f"{N_CPUS}",
            "-i", self.video_path,
            "-pix_fmt", "gray",
            "-f", "rawvideo",
            "pipe:1"
        ]
        pipe = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

        for idx in range(self.num_frames):
            raw = pipe.stdout.read(frame_size)
            if not raw or len(raw) < frame_size:
                self.frames = self.frames[:idx]  # trim the unused part
                break
            self.frames[idx] = np.frombuffer(raw, dtype=np.uint8).reshape((h, w))

        pipe.stdout.close()
        pipe.wait()
        self.num_frames = len(self.frames)

    def load_or_compute(self):
        """
        Loads or computes rotation per frames (if required), threads orientation, and breakpoints for the video.
        """
        if self.calc_rot_per_frame:
            if os.path.isfile(self._dump_path('full_angles')):
                angles = np.load(self._dump_path('full_angles'))
            else:
                angles = self.estimate_threads_orientation(step=1, angle_precision=1800, hough_treshold=200)
                self.dump('full_angles', angles)
            breakpoints = self.compute_breakpoint_candidates(angles, filter_length=8, step=1, merge_threshold=10)
            border_breakpoints = self.filter_breakpoint_candidates(breakpoints, 1)
            self.plot_angles(angles, breakpoint_candidates=breakpoints, breakpoints=border_breakpoints)
            self.rotation_per_frame = self.compute_rotation_per_frame(angles, breakpoints, border_breakpoints)
            logging.debug(f"Calculated rotation per frame\n"
                          f"{self.rotation_per_frame}\n"
                          f"Precalculated rotation per frame\n"
                          f"{ROT_PER_FRAME}\n"
                          f"Difference: {self.rotation_per_frame - ROT_PER_FRAME}\n")
        else:
            self.rotation_per_frame = ROT_PER_FRAME
            logging.debug(f"Loaded precalculated rotation per frame\n"
                          f"{self.rotation_per_frame}\n")

        if os.path.isfile(self._dump_path("angles")):
            self.angles = np.load(self._dump_path("angles"))
            logging.debug(f"Loaded {self._dump_path('angles')}\n")
        else:
            with timing("Compute Angles"):
                self.angles = self.estimate_threads_orientation()
            self.dump("angles", self.angles)

        if os.path.isfile(self._dump_path("breakpoints")) and os.path.isfile(self._dump_path("borderBreakpoints")):
            self.borderBreakpoints = np.load(self._dump_path("borderBreakpoints"))
            logging.debug(f"Loaded {self._dump_path('borderBreakpoints')}\n"
                          f"{self.borderBreakpoints}\n")
            self.breakpoints = np.load(self._dump_path("breakpoints"))
            logging.debug(f"Loaded {self._dump_path('breakpoints')}\n"
                          f"{self.breakpoints}\n")
        else:
            with timing("Compute Breakpoints"):
                self.breakpoints = self.compute_breakpoint_candidates(self.angles)
            self.dump("breakpoints", self.breakpoints)
            with timing("Compute Border Breakpoints"):
                border_breakpoints = self.filter_breakpoint_candidates(self.breakpoints, PREPROCESSOR_SAMPLING)
            with timing("Compute Border Breakpoints"):
                self.borderBreakpoints = self.refine_breakpoints(border_breakpoints, step=PREPROCESSOR_SAMPLING)
            self.dump("borderBreakpoints", self.borderBreakpoints)

        if VERBOSE:
            self.plot_angles(self.angles, breakpoints=self.borderBreakpoints, breakpoint_candidates=self.breakpoints,
                             step=PREPROCESSOR_SAMPLING)

    def estimate_threads_orientation(self, start=0, end=None, step=PREPROCESSOR_SAMPLING, angle_precision=90, hough_treshold=80,
                                     apply_abs=True):
        """
        Estimates thread's orientation for the video frames between start and end with a given step,
        using OpenCV Canny + HoughLinesP.

        Returns:
            np.ndarray: Estimated thread's orientation for each frame.
        """
        computed_angles = []
        if end is None:
            end = self.num_frames

        if LOAD_VIDEO_TO_RAM:
            getFrame = self.getFrameFromRAM
        else:
            getFrame = self.getFrameFromVidCap

        for i in tqdm(range(start, end, step), desc=f"Computing angles from {start} to {end} with step {step}"):
            frame = getFrame(i)

            # Crop and downscale
            frame = frame[Y1:Y2, X1:X2]

            frame = cv2.resize(frame, (
                frame.shape[1] // PREPROCESSOR_DOWNSCALE,
                frame.shape[0] // PREPROCESSOR_DOWNSCALE))
            frame = cv2.GaussianBlur(frame, (5, 5), 1)

            # Edge detection
            edges = cv2.Canny(frame, 70, 180)

            # Hough Transform
            lines = cv2.HoughLinesP(edges, 1, np.pi / angle_precision, hough_treshold,
                                    minLineLength=frame.shape[0] / 2.5, maxLineGap=frame.shape[0] / 3)
            if lines is None:
                logging.debug(f"No lines detected in frame {i}")
                computed_angles.append(45)
                continue

            # Extract angles and compute dominant
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle_rad = np.arctan2(y2 - y1, x2 - x1)
                angles.append(np.rad2deg(angle_rad))

            angle_median = np.median(angles)
            if apply_abs:
                angle_median = np.abs(angle_median)
            computed_angles.append(angle_median)

        logging.debug(f"Angles calculated from {start} to {end} with step {step}\n")
        return np.array(computed_angles)

    def compute_breakpoint_candidates(
            self, orientations=None, filter_length=-1, step=PREPROCESSOR_SAMPLING, threshold=SEGMENT_TYPE_TH,
            merge_threshold=PREPROCESSOR_SAMPLING, secondary=True, segment_type_return=False
    ):
        """
        Computes breakpoint candidates from the thread's orientations.

        Args:
            orientations (np.ndarray): List of angle values. Defaults to self.angles.
            filter_length (int): Filter size for smoothing the angles. Defaults to -1 (no filter).
            step (int): Step size for computation.
            threshold (float): Threshold for detecting changes in segments.
            merge_threshold (int): Threshold for merging close breakpoints.
            secondary (bool): Whether to apply secondary refinement.
            segment_type_return (bool): Whether to return segment types along with breakpoints.

        Returns:
            np.ndarray: Array of breakpoints.
        """
        if orientations is None:
            orientations = self.angles

        polyorder = 1

        # Apply filter if length is valid
        if filter_length >= 2:
            angles_filtered = savgol_filter(orientations, window_length=filter_length, polyorder=polyorder)

            if VERBOSE:
                plt.figure(figsize=(10, 4))
                plt.plot(orientations, label="Original Angles", linestyle="--", alpha=0.7)
                plt.plot(angles_filtered, label="Filtered Angles (Savitzky-Golay)", linestyle="--", alpha=0.7)
                plt.xlabel("Frame Index")
                plt.ylabel("Estimated Angle [°]")
                plt.title("Effect of Savitzky-Golay Filtering on Angle Signal")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_FOLDER, "filter_effect.png"))

                kernel = savgol_coeffs(window_length=filter_length, polyorder=polyorder)
                plt.figure(figsize=(6, 3))
                plt.stem(np.arange(-filter_length // 2 + 1, filter_length // 2 + 1), kernel, basefmt=" ", )
                plt.title("Savitzky–Golay Filter Kernel")
                plt.xlabel("Sample offset")
                plt.ylabel("Weight")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_FOLDER, "sg_filter.png"))

            orientations = angles_filtered

        derivative = np.diff(orientations)
        threshold = threshold * step

        segment_type = np.zeros_like(derivative)
        segment_type[derivative > threshold] = 1  # Increasing
        segment_type[derivative < -threshold] = -1  # Decreasing

        breakpoint_candidates = np.where(np.diff(segment_type) != 0)[0] + 1

        breakpoint_candidates = breakpoint_candidates * step

        if len(breakpoint_candidates) > 1:
            breakpoint_candidates = self.merge_breakpoint_candidates(breakpoint_candidates, step, threshold=merge_threshold, secondary=secondary,
                                                                     segment_type=segment_type)

        breakpoint_candidates = np.concatenate([[0], breakpoint_candidates, [(len(orientations) - 1) * step]])

        logging.debug(f"Calculated: Breakpoint Candidates\n"
                      f"{breakpoint_candidates}\n")

        if segment_type_return:
            return breakpoint_candidates, segment_type
        else:
            self.segment_type = segment_type
            return breakpoint_candidates

    @staticmethod
    def merge_breakpoint_candidates(breakpoint_candidates, step, threshold, segment_type, secondary=True):
        """
        Merges close breakpoint candidates based on a threshold.

        Args:
            breakpoint_candidates (np.ndarray): List of breakpoint candidates.
            step (int): Step size for computation.
            threshold (int): Threshold for merging close breakpoints.
            segment_type (np.ndarray): Segment type of the angles.
            secondary (bool): Whether to apply secondary refinement.

        Returns:
            np.ndarray: Array of merged breakpoint candidates.
        """
        merged_breakpoints = []

        # Temporary group for close breakpoints
        current_group = [breakpoint_candidates[0]]

        # Iterate over the breakpoints
        for i in range(1, len(breakpoint_candidates)):
            # If the difference between consecutive breakpoints is below the threshold, group them
            if breakpoint_candidates[i] - breakpoint_candidates[i - 1] <= threshold:
                current_group.append(breakpoint_candidates[i])
            else:
                # If a current group is finished, calculate the rounded average and store it
                avg_breakpoint = int(round(np.mean(current_group)))
                merged_breakpoints.append(avg_breakpoint)
                # Start a new group
                current_group = [breakpoint_candidates[i]]

        # Handle the last group
        if current_group:
            avg_breakpoint = int(round(np.mean(current_group)))
            merged_breakpoints.append(avg_breakpoint)

        if secondary and len(merged_breakpoints) > 1:
            if stats.mode(segment_type[:merged_breakpoints[0] // step])[0] == \
                    stats.mode(segment_type[merged_breakpoints[0] // step:merged_breakpoints[1] // step])[0]:
                new_merged_breakpoints = []
            else:
                new_merged_breakpoints = [merged_breakpoints[0]]

            for i in range(1, len(merged_breakpoints) - 1):
                if not stats.mode(segment_type[merged_breakpoints[i - 1] // step:merged_breakpoints[i] // step])[
                           0] == \
                       stats.mode(segment_type[merged_breakpoints[i] // step:merged_breakpoints[i + 1] // step])[
                           0]:
                    new_merged_breakpoints.append(merged_breakpoints[i])

            if not stats.mode(segment_type[merged_breakpoints[-1] // step:])[0] == \
                   stats.mode(segment_type[merged_breakpoints[-2] // step:merged_breakpoints[-1] // step])[0]:
                new_merged_breakpoints.append(merged_breakpoints[-1])

            return np.asarray(new_merged_breakpoints)
        else:
            return np.asarray(merged_breakpoints)

    def filter_breakpoint_candidates(self, breakpoint_candidates, step):
        """
        Filter breakpoints from breakpoint candidates based on segments with different trends.

        Args:
            breakpoint_candidates (ndarray): List of breakpoint candidates.
            step (int): Step size for computation.

        Returns:
            np.ndarray: Array of breakpoints.
        """
        breakpoints = []
        for i in range(0, len(breakpoint_candidates) - 1):
            if stats.mode(self.segment_type[breakpoint_candidates[i] // step:breakpoint_candidates[i + 1] // step])[0] == 0:
                breakpoints.append([breakpoint_candidates[i], breakpoint_candidates[i + 1]])

        logging.info(f"Calculated: Breakpoints\n"
                     f"{np.asarray(breakpoints)}\n")

        return np.asarray(breakpoints)

    def refine_breakpoints(self, breakpoints, step):
        """
        Refines the roughly detected breakpoints for greater accuracy.

        Args:
            breakpoints (ndarray): List of rough breakpoints.
            step (int): Step size for computation.

        Returns:
            list: Refined list of breakpoints.
        """
        refined = []

        for start, end in breakpoints:
            if start != 0:
                start_breakpoint = self.refine_breakpoint(start)
            else:
                start_breakpoint = 0

            if end + step < int(self.num_frames):
                end_breakpoint = self.refine_breakpoint(end)
            else:
                end_breakpoint = int(self.num_frames) - 1

            refined.append([start_breakpoint, end_breakpoint])

        logging.info(f"Calculated: Refined Breakpoints\n"
                     f"{refined}\n")

        return refined

    def refine_breakpoint(self, bp):
        """
        Refines a single breakpoint by recomputing angles around it.

        Args:
           bp (int | ndarray): Breakpoint index to refine.

        Returns:
           int: Refined breakpoint index.
        """
        angles = self.estimate_threads_orientation(bp - PREPROCESSOR_SAMPLING, bp + PREPROCESSOR_SAMPLING, 1, 1800, 200)
        breakpoints, segment_type = self.compute_breakpoint_candidates(angles, step=1, merge_threshold=1,
                                                                       segment_type_return=True, filter_length=5)
        if len(breakpoints) == 3:
            refined_bp = breakpoints[1]
        elif len(breakpoints) > 3:
            try:
                refined_bp = self.get_zero_segment(breakpoints, segment_type)
            except Exception as e:
                logging.error(f"Error in refinement: {e}")
                return bp
        else:
            logging.debug(f"No breakpoints found: {breakpoints}")
            return bp
        refined_bp = int(bp - PREPROCESSOR_SAMPLING + refined_bp)
        logging.debug(f"Breakpoint refinement: {bp} -> {refined_bp}")
        return refined_bp

    @staticmethod
    def get_zero_segment(breakpoints, segment_type):
        """
        Identifies the segment closest to zero from the list of breakpoints.

        Args:
            breakpoints (np.ndarray): List of breakpoints.
            segment_type (np.ndarray): Segment type.

        Returns:
            int: Index of the zero segment breakpoint.
        """
        border_breakpoints = []
        for i in range(0, len(breakpoints) - 1):
            if stats.mode(segment_type[breakpoints[i]:breakpoints[i + 1]])[0] == 0:
                border_breakpoints.append([breakpoints[i], breakpoints[i + 1]])

        if len(border_breakpoints) == 1:
            if border_breakpoints[0][0] == 0:
                return border_breakpoints[0][1]
            else:
                return border_breakpoints[0][0]
        else:
            raise Exception("Too many border breakpoints calculated in refinement")

    def preprocess_video(self):
        """
        Applies preprocessing to the video, including cropping, rotating frames and saving the processed output.
        """

        i_row = 0
        if len(self.borderBreakpoints) >= 2:
            start, end = self.borderBreakpoints[i_row]
        else:
            start, end = 0, 999999

        angle = float(self.estimate_threads_orientation(start, start + 1, 1, 1800, 200, False)[0]) + PITCH_ANGLE

        if LOAD_VIDEO_TO_RAM:
            getFrame = self.getFrameFromRAM
            setFrame = self.setFrameToRAM
        else:
            getFrame = self.getFrameFromVidCap
            setFrame = self.setFrameToVidCap
            self.video_capture = cv2.VideoCapture(self.video_path)
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.video_writer = cv2.VideoWriter(str(self.get_output_video_file_path()),
                                                apiPreference=cv2.CAP_FFMPEG,
                                                fourcc=cv2.VideoWriter_fourcc(*CODEC),
                                                fps=self.video_capture.get(cv2.CAP_PROP_FPS),
                                                frameSize=(X2 - X1, Y2 - Y1),
                                                params=[
                                                    cv2.VIDEOWRITER_PROP_DEPTH,
                                                    cv2.CV_8U,
                                                    cv2.VIDEOWRITER_PROP_IS_COLOR,
                                                    0,
                                                ]
                                                )

        time_read = 0
        time_remap = 0
        time_rotation = 0
        time_write = 0

        for i in tqdm(range(self.num_frames), desc="PreProcessing frames"):
            start_time = time.time()
            frame = getFrame(i)
            time_read += time.time() - start_time
            if RECTIFY:
                start_time = time.time()
                frame = cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_LINEAR).astype(np.uint8)
                time_remap += time.time() - start_time

            frame = frame[Y1 - PADDING:Y2 + PADDING, X1 - PADDING:X2 + PADDING]

            rotate_matrix = cv2.getRotationMatrix2D((frame.shape[1] / 2, frame.shape[0] / 2), angle, 1)

            start_time = time.time()
            rotated_image = cv2.warpAffine(
                src=frame,
                M=rotate_matrix,
                dsize=(frame.shape[1], frame.shape[0]),
                flags=cv2.INTER_CUBIC
            )[PADDING:Y2 - Y1 + PADDING, PADDING:X2 - X1 + PADDING]
            time_rotation += time.time() - start_time

            start_time = time.time()
            setFrame(rotated_image.astype(np.uint8), i)
            time_write += time.time() - start_time

            if i < start:
                angle += self.rotation_per_frame
            if i == end:
                i_row += 1
                if i_row < len(self.borderBreakpoints):
                    start, end = self.borderBreakpoints[i_row]
                else:
                    logging.warning(f"Reached out of index {i_row}")

        logging.debug(f"read() {time_read}\n")
        logging.debug(f"remap() {time_remap}\n")
        logging.debug(f"warpAffine() {time_rotation}\n")
        logging.debug(f"write() {time_write}\n")

    def compute_rotation_per_frame(self, orientations, breakpoint_candidates, breakpoints):
        """
        Computes the rotation per frame based on thread's orientations, breakpoint candidates, and breakpoints.

        Args:
            orientations (list): List of estimated thread's orientations.
            breakpoint_candidates (np.ndarray): Detected breakpoint candidates.
            breakpoints (np.ndarray): Detected breakpoints.

        Returns:
            float: Rotation per frame.
        """
        fa = []
        j = 0
        k = 0
        start, end = breakpoints[j]
        for i in range(0, len(breakpoint_candidates) - 1):
            if not start == breakpoint_candidates[i]:
                segment = stats.mode(self.segment_type[breakpoint_candidates[i]:breakpoint_candidates[i + 1]])[0]
                offset = -(k * 180)
                if segment == 1 and not start == breakpoint_candidates[i + 1]:
                    k += 1
                f = orientations[breakpoint_candidates[i]:breakpoint_candidates[i + 1]] * -segment + offset
                fa = np.concatenate([fa, f])
            else:
                j += 1
                if j < len(breakpoints):
                    start, end = breakpoints[j]

        a, b = np.polyfit(np.arange(len(fa)), fa, 1)

        plt.figure(figsize=(10, 4))
        plt.plot(fa, label="Cumulative Rotation Angles", color="blue")
        plt.plot([(a * x + b) for x in np.arange(len(fa))], label="Linear Fit (Polyfit)", color="red", linestyle="--")
        plt.xlabel("Frame Index")
        plt.ylabel("Rotation Angle (degrees)")
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.savefig(self._dump_path("RPF_function", "png"))
        plt.close()

        logging.info(f"Calculated : Angular Rotation Per Frame\n"
                     f"{-a}")

        return -a

    def get_intervals(self):
        """
        Calculates intervals between breakpoints of segments with horizontal movement.

        Returns:
            list: List of filtered intervals.
        """
        start = 0
        end = self.num_frames
        inverted = []

        # Step 1: Generate inverted intervals
        if self.borderBreakpoints[0][0] > start:
            inverted.append([start, self.borderBreakpoints[0][0] - 1])

        for i in range(1, len(self.borderBreakpoints)):
            inverted.append([self.borderBreakpoints[i - 1][1] + 1, self.borderBreakpoints[i][0] - 1])

        if self.borderBreakpoints[-1][1] < end:
            inverted.append([self.borderBreakpoints[-1][1] + 1, end])

        # Step 2: Calculate average size of intervals
        sizes = [interval[1] - interval[0] + 1 for interval in inverted]
        avg_size = sum(sizes) / len(sizes)

        # Step 3: Define a threshold
        threshold = INTERVAL_FILTER_TH * avg_size

        # Step 4: Filter first and last intervals if they are below the threshold
        if len(inverted) > 1 and (inverted[0][1] - inverted[0][0] + 1) < threshold:
            inverted.pop(0)
        if len(inverted) > 1 and (inverted[-1][1] - inverted[-1][0] + 1) < threshold:
            inverted.pop(-1)

        return inverted

    def plot_angles(self, orientations, breakpoint_candidates=None, breakpoints=None, step=1):
        """
        Plots thread's orientations and optionally marks breakpoints and breakpoint candidates.

        Args:
            orientations (np.ndarray): List of thread's orientations to plot.
            breakpoint_candidates (np.ndarray, optional): List of breakpoint candidates to mark on the plot.
            breakpoints (list, optional): List of breakpoints to mark on the plot.
            step (int, optional): Step size of thread's orientation estimation.
        """
        plt.figure(figsize=(15, 3))
        x = range(len(orientations))
        x = [i * step for i in x]
        plt.plot(x, orientations)
        if breakpoint_candidates is not None:
            for bp in breakpoint_candidates:
                plt.axvline(bp, color="blue")
        if breakpoints is not None:
            for start, end in self.borderBreakpoints:
                plt.axvline(start, color="green")
                plt.axvline(end, color="red")
        plt.show()
        plt.savefig(self._dump_path("angles", "png"))
        plt.close()

    def setFrameToRAM(self, frame, i):
        self.processed_frames[i] = frame.astype(np.uint8)

    def setFrameToVidCap(self, frame, i):
        self.video_writer.write(frame.astype(np.uint8))

    def getProcessedFrames(self):
        return self.processed_frames

    def get_output_video_file_path(self):
        """
        Retrieves the path for the output processed video file.

        Returns:
            str: Path to the output video file.
        """
        return self.output_video_file_path

    def getFrameFromRAM(self, i):
        return self.frames[i]

    def getFrameFromVidCap(self, i):
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, frame = self.video_capture.read()

        if not success or frame is None:
            logging.critical(f"Failed to read frame {i}")
            raise IOError(f"Failed to read frame {i}")

        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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

    def dump(self, name: str, obj):
        """
        Saves an object to a file using numpy save function.

        Args:
            name (str): Name of the object.
            obj: The object to save.
        """
        np.save(self._dump_path(name), obj)
