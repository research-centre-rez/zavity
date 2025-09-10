import logging
import os.path
import pickle
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from config.config import (MOTION_DOWNSCALE, ROW_ROTATION_OVERLAP_RATIO, OUTPUT_FOLDER, VERBOSE)
import multiprocessing


class VideoMotion:
    """
    Handles motion detection and speed calculation. This means:
    - estimating shift of each pair of consecutive frames
    - postprocessing of raw values

    Attributes:
        speeds (dict[str, float]): Dictionary containing horizontal and vertical speeds.
        stats (dict): Statistical data for speed calculations.
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
    motion_positions: list[tuple[int, float, float]]
    intervals: np.ndarray
    width: int
    height: int
    video_file_path: str
    video_name: str
    frames_per_360: int
    cw: bool
    frames: np.ndarray

    def __init__(self, video_file_path: str, intervals: list):
        """
        Initializes the VideoMotion class.

        Args:
            frames (np.ndarray): Array of frames used when processing in RAM.
            video_file_path (str): Path to the input video file.
            intervals (list): List of intervals with vertical computed during video preprocessing.
        """
        self.speeds = {}
        self.stats = {}
        self.motion_directions = []
        self.motion_positions = []
        self.intervals = np.array(intervals)

        video_capture = cv2.VideoCapture(video_file_path)
        self.width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH) / MOTION_DOWNSCALE)
        self.height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT) / MOTION_DOWNSCALE)
        self.num_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        video_capture.release()

        self.video_file_path = video_file_path
        self.video_name = os.path.basename(video_file_path)

    def process(self):
        """
        Processes video motion data by estimating (storing)/loading motion-related attributes.
        Checks for precomputed motion attributes and attempts to load them if available. If not, it calculates
        and persists the attributes for subsequent use. Also handles loading and logging of motion statistics.

        Raises:
            Any exceptions related to file operations, computations, or data processing.
        """
        logging.info(f"Processing VideoMotion for: {self.video_file_path}\n")
        if os.path.isfile(self._dump_path("motion_directions")) and os.path.isfile(self._dump_path("motion_positions")):
            self.motion_directions = np.load(self._dump_path("motion_directions"))
            self.motion_positions = np.load(self._dump_path("motion_positions"))
        else:
            self.parallel_motion_est()
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
            self._compute()

    @staticmethod
    def sequential_motion_est(params):
        """
        Estimates the motion between frames in a specified range of a video file using feature tracking.

        This function calculates the motion of features in consecutive frames using the Lucas-Kanade optical flow
        method. It tracks features in the given video file, starting from a specific frame and ending at another.
        The motion is represented as a series of positions for each frame processed.

        Arguments:
            params (tuple): A tuple containing:
                video_file_path (str): Path to the video file.
                start (int): Starting frame number of the range.
                end (int): Ending frame number of the range.

        Returns:
            tuple: A tuple containing:
                - motion_positions (list of tuple): A list where each entry is a tuple consisting of the frame number,
                  cumulative horizontal position, and cumulative vertical position.
        """
        video_file_path, start, end = params

        feature_params = dict(maxCorners=200,
                              qualityLevel=0.1,
                              minDistance=7,
                              blockSize=7)

        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        vidcap = cv2.VideoCapture(video_file_path)
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)  # This operation is time consuming and must be done rarely
        success, prev_frame = vidcap.read()
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(prev_frame, **feature_params)

        motion_positions = [(0, 0.0, 0.0)]

        for frame_no in tqdm(np.arange(start + 1, end), total=end - start - 1, desc=f"Motion for frames {start}-{end}"):
            success, next_frame = vidcap.read()
            if not success:
                break
            next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
            new_corners, status, err = cv2.calcOpticalFlowPyrLK(prev_frame, next_frame, corners, None, **lk_params)
            good_old = corners[status == 1]
            good_new = new_corners[status == 1]
            movement_direction = np.median(good_new - good_old, axis=0)
            motion_positions.append((frame_no, motion_positions[-1][1] + movement_direction[0],
                                     motion_positions[-1][2] + movement_direction[1]))


            corners = new_corners[status == 1]
            if len(corners < 150):
                prev_frame = next_frame
                corners = cv2.goodFeaturesToTrack(prev_frame, **feature_params)

        vidcap.release()
        return motion_positions

    def parallel_motion_est(self):
        """
        Compute and aggregate motion trajectories for video frames using multiprocessing framework.

        This function employs a multiprocessing pool to compute motion estimates for distinct
        video frame sequences. Each sequence's result is aggregated to construct the overall
        motion trajectory while maintaining continuity across sequences. The method ensures
        efficient parallel computation, accommodating systems with multiple CPU cores.

        Attributes
        ----------
        motion_positions: numpy.ndarray
            Aggregated array representing the motion trajectory of the video frames, aligned
            and merged across different computed sequences.

        """
        cpus = multiprocessing.cpu_count() - 1
        with multiprocessing.Pool(cpus) as pool:
            results = list(tqdm(pool.imap(VideoMotion.sequential_motion_est,
                                          [(self.video_file_path, np.max([frame_no - 1, 0]), frame_no + self.num_frames // cpus)
                                           for frame_no in np.arange(0, self.num_frames, self.num_frames // cpus)]),
                                total=cpus,  # (end-start)/step,
                                desc=f"Computing motion")
                           )

        # aggregate measured results
        stack = []
        dx = dy = 0  # each sequence is necessary to shift according to the previous sequence
        for pos in results:
            pos_a = np.array(pos)
            start, _, _ = pos[1]
            end, x1, y1 = pos[-1]
            if start == 0:  #first sequence contains (0, 0, 0) for the other sequences it is redundant
                stack.append(np.stack([pos_a[:, 0], pos_a[:, 1], pos_a[:, 2]], axis=1))
            else:
                stack.append(np.stack([pos_a[1:, 0], pos_a[1:, 1] + dx, pos_a[1:, 2] + dy], axis=1))
            dx += x1
            dy += y1
        self.motion_positions = np.concatenate(stack)

        if VERBOSE:
            self.plot_motion_trajectory()


    def _compute(self):
        """
        Performs computational operations involving speeds, intervals, frames, and statistical
        data. The method orchestrates the computation and stores data in various formats
        for later use.

        Raises:
            Exception: If there are issues in dumping or file operations.
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

        frame_shift_estimate = int(np.ceil(np.median(self.intervals[:, 1] - self.intervals[:, 0]) / ROW_ROTATION_OVERLAP_RATIO))

        feature_params = dict(maxCorners=50,
                              qualityLevel=0.1,
                              minDistance=50,
                              blockSize=7)

        lk_params = dict(winSize=(50, 50),
                         maxLevel=3,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.1))

        vidcap = cv2.VideoCapture(self.video_file_path)
        for start, end in tqdm(self.intervals, total=len(self.intervals), desc="Counting row lengths"):
            samples = []

            # Reference frame for matching points:
            for i in range(-40, 41, 20):
                ref_frame = (end - start - frame_shift_estimate) // 2 + start + i
                vidcap.set(cv2.CAP_PROP_POS_FRAMES, ref_frame)
                success, a = vidcap.read()
                a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
                corners = cv2.goodFeaturesToTrack(a, **feature_params)

                # in the for loop try several frames to match founded corners
                ROT360_SEARCH_RANGE = 100
                vidcap.set(cv2.CAP_PROP_POS_FRAMES, ref_frame + frame_shift_estimate - ROT360_SEARCH_RANGE)
                matches = []
                for frame_shift_delta in range(-ROT360_SEARCH_RANGE, ROT360_SEARCH_RANGE):
                    success, b = vidcap.read()
                    if not success:
                        continue
                    b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
                    p1, st, err = cv2.calcOpticalFlowPyrLK(a, b, corners, None, **lk_params)
                    p1 = p1[st == 1]
                    p0 = corners[st == 1]

                    match_quality = np.sum(err[st == 1])
                    move = np.median(p1 - p0, axis=0)
                    matches.append((match_quality, move[0], frame_shift_estimate + frame_shift_delta))

                best_match_id = np.argmin(np.array(matches)[:, 0])
                if self.get_direction() == 'CW':
                    samples.append((np.min(np.array(matches)[:, 0]), matches[best_match_id][1:], matches[best_match_id][2]))
                else:
                    samples.append((np.min(np.array(matches)[:, 0]), -matches[best_match_id][1], matches[best_match_id][2]))

            results.append(np.median([frames_count for _, _, frames_count in samples]))

        self.frames_per_360 = np.median(results)
        std_over_rows = np.std(results) / abs(self.speeds['horizontal'])
        logging.debug(
            f"Frames per 360: {self.frames_per_360}±{std_over_rows} calculated from frame_shift: {frame_shift_estimate}")

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
