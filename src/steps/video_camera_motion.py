import logging
import os.path
import pickle
import cv2
import numpy as np
import pandas as pd
from scipy.stats import stats
from tqdm.auto import tqdm

from config.config import (N_CPUS, MOTION_SAMPLING, MOTION_DOWNSCALE, ROW_ROTATION_OVERLAP_RATIO, LOAD_VIDEO_TO_RAM,
                           OUTPUT_FOLDER, VERBOSE)


class VideoMotion:
    """
    Handles motion detection and speed calculation.

    Attributes:
        speeds (dict[str, float]): Dictionary containing horizontal and vertical speeds.
        stats (dict): Statistical data for speed calculations.
        motion_directions (list[int]): List of motion directions for frames.
        motion_positions (list[tuple[int, float, float]]): Position data for each frame.
        intervals (np.ndarray): Intervals of motion detected in the video.
        video_capture (cv2.VideoCapture): OpenCV video capture object.
        width (int): Width of the resized video frame.
        height (int): Height of the resized video frame.
        video_file_path (str): Path to the input video file.
        video_name (str): Name of the video file.
        frames_per_360 (int): Number of frames for a 360-degree rotation.
        cw (bool): Indicates whether the motion is clockwise.
        frames (np.ndarray): Array of frames used when processing in RAM.
    """
    speeds: dict[str, float]
    stats: dict
    motion_directions: list[int]
    motion_positions: list[tuple[int, float, float]]
    intervals: np.ndarray
    video_capture: cv2.VideoCapture
    width: int
    height: int
    video_file_path: str
    video_name: str
    frames_per_360: int
    cw: bool
    frames: np.ndarray

    def __init__(self, frames: np.ndarray, video_file_path: str, intervals: list):
        """
        Initializes the VideoMotion class.

        Args:
            frames (np.ndarray): Array of frames used when processing in RAM.
            video_file_path (str): Path to the input video file.
            intervals (list): List of intervals with vertical computed during video preprocessing.
        """
        cv2.setNumThreads(N_CPUS)
        self.speeds = {}
        self.stats = {}
        self.motion_directions = []
        self.motion_positions = []
        self.intervals = np.array(intervals)
        if LOAD_VIDEO_TO_RAM:
            self.frames = frames
            self.width, self.height = frames.shape[2] / MOTION_DOWNSCALE, frames.shape[1] / MOTION_DOWNSCALE
        else:
            self.video_capture = cv2.VideoCapture(video_file_path)
            self.width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH) / MOTION_DOWNSCALE)
            self.height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT) / MOTION_DOWNSCALE)
        self.video_file_path = video_file_path
        self.video_name = os.path.basename(video_file_path)

    def process(self):
        """
        Processes motion analysis for the video.
        """
        logging.info(f"Processing VideoMotion for: {self.video_file_path}\n")
        self.load_or_compute()

    def load_or_compute(self):
        """
        Loads or computes motion data, speeds, and intervals.
        """
        if os.path.isfile(self._dump_path("motion_directions")) and os.path.isfile(self._dump_path("motion_positions")):
            self.motion_directions = np.load(self._dump_path("motion_directions"))
            self.motion_positions = np.load(self._dump_path("motion_positions"))
        else:
            self.compute_motion()
            self.dump("motion_directions", self.motion_directions)
            self.dump("motion_positions", self.motion_positions)
        if os.path.isfile(self._dump_path("speeds")) and os.path.isfile(
                self._dump_path("frames_per_360")) and os.path.isfile(self._dump_path("stats")):
            with open(self._dump_path("speeds"), 'rb') as fp:
                self.speeds = pickle.load(fp)
            with open(self._dump_path("stats"), 'rb') as fp:
                self.stats = pickle.load(fp)
            logging.debug(f"Horizontal speed: {self.speeds['horizontal']}±{self.stats['horizontal_speed_std']}\n"
                         f"Vertical shift: {self.speeds['vertical_shift']}±{self.stats['vertical_shift_std']}\n"
                         f"Clockwise: {self.get_direction()}\n"
                         f"Moving down: {self.is_moving_down()}\nLoaded\n")
            self.frames_per_360 = np.load(self._dump_path("frames_per_360"))
            logging.debug(f"Frames per 360: {self.frames_per_360} Loaded")
        else:
            self.compute()

    def compute_motion(self):
        """
        Computes motion directions and positions for each frame in the video using Optical Flow.
        """
        feature_params = dict(maxCorners=100,
                              qualityLevel=0.1,
                              minDistance=7,
                              blockSize=7)

        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        err_threshold = 9

        if LOAD_VIDEO_TO_RAM:
            getFrame = self.getFrameFromRAM
            total_frames = len(self.frames)
        else:
            getFrame = self.getFrameFromVidCap
            total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        prev_frame = getFrame(0)
        prev_frame = cv2.resize(prev_frame,
                                (prev_frame.shape[1] // MOTION_DOWNSCALE, prev_frame.shape[0] // MOTION_DOWNSCALE))
        self.motion_positions.append((0, 0.0, 0.0))

        for i in tqdm(range(MOTION_SAMPLING, total_frames, MOTION_SAMPLING), desc=f"Processing motion from frames"):
            frame = getFrame(i)

            frame = cv2.resize(frame, (frame.shape[1] // MOTION_DOWNSCALE, frame.shape[0] // MOTION_DOWNSCALE))
            corners = self.get_corners(prev_frame, **feature_params)

            if corners is not None:
                p1, st, err = cv2.calcOpticalFlowPyrLK(prev_frame, frame, corners, None, **lk_params)

                st = (st == 1) & (err < err_threshold)
                good_new = p1[st == 1]
                good_old = corners[st == 1]
                if good_new.shape[0] > 0:
                    movement_direction = np.median(good_new - good_old, axis=0)
                    max_pos = np.argmax(np.abs(movement_direction))
                    self.motion_positions.append((i,
                                                  self.motion_positions[-1][1] + movement_direction[0],
                                                  self.motion_positions[-1][2] + movement_direction[1]))

                    if max_pos == 0:
                        if movement_direction[max_pos] > 0:
                            self.motion_directions.append(1)
                        else:
                            self.motion_directions.append(2)
                    else:
                        if movement_direction[max_pos] > 0:
                            self.motion_directions.append(3)
                        else:
                            self.motion_directions.append(4)
                else:
                    self.motion_positions.append((i,
                                                  self.motion_positions[-1][1] + self.motion_positions[-1][1] -
                                                  self.motion_positions[-2][1],
                                                  self.motion_positions[-1][2] + self.motion_positions[-1][2] -
                                                  self.motion_positions[-2][2]))
                    self.motion_directions.append(0)
            else:
                self.motion_positions.append((i,
                                              self.motion_positions[-1][1] + self.motion_positions[-1][1] -
                                              self.motion_positions[-2][1],
                                              self.motion_positions[-1][2] + self.motion_positions[-1][2] -
                                              self.motion_positions[-2][2]))
                self.motion_directions.append(0)

            # Now update the previous frame and previous points
            prev_frame = frame

        if VERBOSE:
            self.plot_motion_trajectory()

    def compute(self):
        """
        Computes speeds, intervals, and frames per 360-degree rotation.
        """
        self.compute_speeds()
        self.compute_frames_per360()
        self.dump("intervals", self.intervals)
        with open(self._dump_path("speeds"), 'wb') as fp:
            pickle.dump(self.speeds, fp)
        with open(self._dump_path("stats"), 'wb') as fp:
            pickle.dump(self.stats, fp)
        self.dump("frames_per_360", self.frames_per_360)

    def compute_speeds(self):
        """
        Computes horizontal and vertical speeds from the detected motion.
        """
        columns = ["frame_ID", "x_shift", "y_shift"]
        df = pd.DataFrame(self.motion_positions, columns=columns)
        df["x_shift_diff"] = df["x_shift"].diff()
        df.loc[0, "x_shift_diff"] = df["x_shift"].iloc[0]
        df["y_shift_diff"] = df["y_shift"].diff()
        df.loc[0, "y_shift_diff"] = df["y_shift"].iloc[0]

        mask_horizontal = pd.Series(False, index=df.index)
        for start, end in self.intervals:
            mask_horizontal |= (df['frame_ID'] >= start + 5) & (df['frame_ID'] <= end - 5)
        df_horizontal = df[mask_horizontal]
        self.speeds["horizontal"] = df_horizontal["x_shift_diff"].mean() * MOTION_DOWNSCALE
        self.stats["horizontal_speed_std"] = df_horizontal["x_shift_diff"].std() * MOTION_DOWNSCALE

        # mask_vertical = pd.Series(False, index=df.index)
        # for start, end in self.getInvertedIntervals():
        #     mask_vertical |= (df['frame_ID'] >= start+5) & (df['frame_ID'] <= end-5)
        # df_vertical = df[mask_vertical]
        # self.speeds["vertical"] = df_vertical["y_shift_diff"].mean() * self.resolution_decs

        vertical_shifts = []
        for start, end in self.get_inverted_intervals():
            vertical_shift = df.loc[end, "y_shift"] - df.loc[start, "y_shift"]
            vertical_shifts.append(vertical_shift)
        self.speeds["vertical_shift"] = np.mean(vertical_shifts) * MOTION_DOWNSCALE
        self.stats["vertical_shift_std"] = np.std(vertical_shifts) * MOTION_DOWNSCALE

        logging.debug(
            f"Horizontal speed: {self.speeds['horizontal']}±{self.stats['horizontal_speed_std']}\n"
            f"Vertical shift: {self.speeds['vertical_shift']}±{self.stats['vertical_shift_std']}\n"
            f"Clockwise: {self.get_direction()}\n"
            f"Moving down: {self.is_moving_down()}\nCalculated\n")

    def compute_frames_per360(self):
        """
        Calculates the number of frames required for a 360-degree rotation using Optical Flow.
        """
        results = []

        if LOAD_VIDEO_TO_RAM:
            getFrame = self.getFrameFromRAM
        else:
            getFrame = self.getFrameFromVidCap

        frame_shift = int(np.ceil(np.mean(self.intervals[:, 1] - self.intervals[:, 0]) / ROW_ROTATION_OVERLAP_RATIO))

        for start, end in self.intervals:
            samples = []
            for i in range(20, 101, 20):
                frame = int(start + i)
                a = getFrame(frame)
                b = getFrame(frame + frame_shift)

                feature_params = dict(maxCorners=50,
                                      qualityLevel=0.1,
                                      minDistance=50,
                                      blockSize=7)

                lk_params = dict(winSize=(50, 50),
                                 maxLevel=3,
                                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.1))

                err_threshold = 9

                corners = cv2.goodFeaturesToTrack(a, **feature_params)

                p1, st, err = cv2.calcOpticalFlowPyrLK(a, b, corners, None, **lk_params)

                p1 = p1[st == 1]
                p0 = corners[st == 1]

                move = np.median(p1 - p0, axis=0)

                if self.get_direction() == 'CW':
                    samples.append(move[0])
                else:
                    samples.append(-move[0])

            results.append(np.median(samples))

        result = np.mean(results)
        self.frames_per_360 = np.ceil(frame_shift + result / abs(self.speeds['horizontal'])).astype(int)
        std_over_rows = np.std(results) / abs(self.speeds['horizontal'])
        logging.debug(
            f"Frames per 360: {frame_shift + result / abs(self.speeds['horizontal'])}±{std_over_rows} calculated from "
            f"frame_shift: {frame_shift}")

    @staticmethod
    def get_corners(gray_frame, **feature_params):
        """
        Detects corners in a grayscale frame using OpenCV's goodFeaturesToTrack.

        Args:
            gray_frame (np.ndarray): Grayscale image for corner detection.
            **feature_params: Additional parameters for corner detection.

        Returns:
            np.ndarray: Detected corner points.
        """
        corners = cv2.goodFeaturesToTrack(gray_frame, **feature_params)
        return corners

    def _dump_path(self, obj_name):
        """
        Generates a file path for saving or loading objects.

        Args:
            obj_name (str): Name of the object to save/load.

        Returns:
            str: Full path to the file.
        """
        return os.path.join(OUTPUT_FOLDER, os.path.splitext(self.video_name)[0] + f'-{obj_name}.npy')

    def dump(self, name: str, obj):
        """
        Saves an object as a NumPy file.

        Args:
            name (str): Name of the object.
            obj: Object to save.
        """
        np.save(self._dump_path(name), obj)

    @staticmethod
    def get_common_direction(motion_direction):
        """
        Finds the most frequent motion direction.

        Args:
            motion_direction (list[int]): List of motion directions.

        Returns:
            int: Most common motion direction.
        """
        return stats.mode(motion_direction)

    def getFrameFromRAM(self, i):
        return self.frames[i]

    def getFrameFromVidCap(self, i):
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, frame = self.video_capture.read()

        if not success or frame is None:
            logging.critical(f"Failed to read frame {i}")
            raise IOError(f"Failed to read frame {i}")

        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def plot_motion_trajectory(self, start=0, end=4000):
        """
        Plots the cumulative horizontal and vertical motion over time
        and saves the result to the specified path.
        """
        if not hasattr(self, "motion_positions") or len(self.motion_positions) == 0:
            logging.warning("No motion data found. Run compute_motion() first.")
            return

        # Extract frame indices and displacements
        frame_indices = [entry[0] for entry in self.motion_positions[:4000]]
        horizontal_shifts = [entry[1] for entry in self.motion_positions[:4000]]
        vertical_shifts = [entry[2] for entry in self.motion_positions[:4000]]

        # Plot
        from matplotlib import pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.plot(frame_indices, horizontal_shifts, label="Horizontal Displacement", color="blue")
        plt.plot(frame_indices, vertical_shifts, label="Vertical Displacement", color="red")
        plt.xlabel("Frame Index")
        plt.ylabel("Cumulative Displacement (pixels)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FOLDER, "motion.png"))
        plt.close()

    def get_intervals(self):
        """
        Retrieves motion intervals.

        Returns:
            np.ndarray: Detected intervals.
        """
        return self.intervals

    def get_horizontal_speed(self):
        """
        Retrieves the absolute horizontal speed.

        Returns:
            float: Absolute value of horizontal speed.
        """
        return abs(self.speeds['horizontal'])

    def get_vertical_speed(self):
        """
        Retrieves the absolute vertical speed.

        Returns:
            float: Absolute value of vertical speed.
        """
        return abs(self.speeds['vertical'])

    def get_average_vertical_shift(self):
        """
        Retrieves the average vertical shift across motion intervals.

        Returns:
            float: Average vertical shift.
        """
        return abs(self.speeds['vertical_shift'])

    def is_moving_down(self):
        """
        Determines if the motion is moving downward.

        Returns:
            bool: True if the motion is downward, False otherwise.
        """
        return self.speeds['vertical_shift'] < 0

    def get_direction(self):
        """
        Identifies the direction of the motion.

        Returns:
            str: "CCW" (counter-clockwise) or "CW" (clockwise).
        """
        return "CCW" if self.speeds['horizontal'] > 0 else "CW"

    def get_inverted_intervals(self):
        """
        Computes intervals that are inverted.

        Returns:
            np.ndarray: Array of inverted intervals.
        """
        return np.array([[self.intervals[i - 1][1], self.intervals[i][0]] for i in range(1, len(self.intervals))])

    def get_average_horizontal_shift(self):
        return np.mean(self.intervals[:, 1] - self.intervals[:, 0]) * self.get_horizontal_speed()

    def get_frames_per360(self):
        """
        Retrieves the number of frames required for a 360-degree rotation.

        Returns:
            int: Frames per 360-degree rotation.
        """
        return self.frames_per_360
