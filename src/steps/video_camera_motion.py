import os.path
import pickle
import cv2
import numpy as np
import pandas as pd
from scipy.stats import stats
from tqdm.auto import tqdm

from config.config import FPS_REDUCTION, RESOLUTION_DECS, ROW_OVERLAP


class VideoMotion:
    """
    Handles motion detection and speed calculation for video preprocessing.

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
        output_path (str): Path to store output results.
        video_name (str): Name of the video file.
        frames_per_360 (int): Number of frames for a 360-degree rotation.
        cw (bool): Indicates whether the motion is clockwise.
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
    output_path: str
    video_name: str
    frames_per_360 = int
    cw = bool

    def __init__(self, video_file_path: str, output_path: str, intervals: list):
        """
        Initializes the VideoMotion class.

        Args:
            video_file_path (str): Path to the input video file.
            output_path (str): Directory to save outputs.
            intervals (list): List of intervals with vertical computed during video preprocessing.
        """
        self.speeds = {}
        self.stats = {}
        self.motion_directions = []
        self.motion_positions = []
        self.intervals = np.array(intervals)
        self.video_capture = cv2.VideoCapture(video_file_path)
        self.width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH) / RESOLUTION_DECS)
        self.height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT) / RESOLUTION_DECS)
        self.video_file_path = video_file_path
        self.output_path = output_path
        self.video_name = os.path.basename(video_file_path)

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
        return os.path.join(self.output_path, os.path.splitext(self.video_name)[0] + f'-{obj_name}.npy')

    def dump(self, name: str, obj):
        """
        Saves an object as a NumPy file.

        Args:
            name (str): Name of the object.
            obj: Object to save.
        """
        np.save(self._dump_path(name), obj)

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
        if os.path.isfile(self._dump_path("speeds")) and os.path.isfile(self._dump_path("frames_per_360")) and os.path.isfile(self._dump_path("stats")):
            with open(self._dump_path("speeds"), 'rb') as fp:
                self.speeds = pickle.load(fp)
            with open(self._dump_path("stats"), 'rb') as fp:
                self.stats = pickle.load(fp)
            print(f"Horizontal speed: {self.speeds['horizontal']}±{self.stats['horizontal_speed_std']}\n"
                  f"Vertical shift: {self.speeds['vertical_shift']}±{self.stats['vertical_shift_std']}\n"
                  f"Clockwise: {self.get_direction()}, Portrait: {self.is_portrait()}\nLoaded\n")
            self.frames_per_360 = np.load(self._dump_path("frames_per_360"))
            print(f"Frames per 360: {self.frames_per_360} Loaded")
        else:
            self.compute()

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

    def process(self):
        """
        Processes motion analysis for the video.
        """
        print(f"Processing VideoMotion for: {self.video_file_path}\n")
        self.load_or_compute()

    def compute_motion(self):
        """
        Computes motion directions and positions for each frame in the video.
        """
        feature_params = dict(maxCorners=100,
                              qualityLevel=0.1,
                              minDistance=7,
                              blockSize=7)

        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        err_threshold = 9

        ret, old_frame = self.video_capture.read()
        old_gray = cv2.cvtColor(cv2.resize(old_frame, (self.width, self.height)), cv2.COLOR_BGR2GRAY)

        self.motion_positions.append((int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)), 0.0, 0.0))

        total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=total_frames // FPS_REDUCTION, desc='Processing motion from frames')

        # current_frame_n = self.fps_reduction

        while True:
            # self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame_n)
            ret, frame = self.video_capture.read()
            if not ret:
                break

            frame_gray = cv2.cvtColor(cv2.resize(frame, (self.width, self.height)), cv2.COLOR_BGR2GRAY)
            corners = self.get_corners(old_gray, **feature_params)

            if corners is not None:
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, corners, None, **lk_params)

                st = (st == 1) & (err < err_threshold)
                good_new = p1[st == 1]
                good_old = corners[st == 1]
                if good_new.shape[0] > 0:
                    movement_direction = np.mean(good_new - good_old, axis=0)
                    max_pos = np.argmax(np.abs(movement_direction))
                    self.motion_positions.append((int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)),
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
                    self.motion_positions.append((int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)),
                                                  self.motion_positions[-1][1] + self.motion_positions[-1][1] -
                                                  self.motion_positions[-2][1],
                                                  self.motion_positions[-1][2] + self.motion_positions[-1][2] -
                                                  self.motion_positions[-2][2]))
                    self.motion_directions.append(0)
            else:
                self.motion_positions.append((int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)),
                                              self.motion_positions[-1][1] + self.motion_positions[-1][1] -
                                              self.motion_positions[-2][1],
                                              self.motion_positions[-1][2] + self.motion_positions[-1][2] -
                                              self.motion_positions[-2][2]))
                self.motion_directions.append(0)

            # Now update the previous frame and previous points
            old_gray = frame_gray

            # Update the progress bar
            pbar.update(1)
            # current_frame_n += self.fps_reduction

        cv2.destroyAllWindows()
        pbar.close()

    def show_frames(self, frame_numbers):
        """
        Displays specified frames from the video for visualization.

        Args:
            frame_numbers (list[int]): List of frame indices to display.
        """
        for frame_number in frame_numbers:
            # Set the position of the video to the desired frame
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            # Read the frame
            ret, frame = self.video_capture.read()

            height, width = frame.shape[:2]

            # Resize the frame
            new_width = int(width * 0.3)
            new_height = int(height * 0.3)
            resized_frame = cv2.resize(frame, (new_width, new_height))

            # Display the frame
            cv2.imshow(f"Frame {frame_number}", resized_frame)

        # Wait for a key press and close the display window
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Release the video capture object
        self.video_capture.release()

    def get_intervals(self):
        """
        Retrieves motion intervals.

        Returns:
            np.ndarray: Detected intervals.
        """
        return self.intervals

    def compute_speeds(self):
        """
        Computes horizontal and vertical speeds from the detected motion.
        """
        if self.is_portrait():
            columns = ["frame_ID", "y_shift", "x_shift"]
        else:
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
        self.speeds["horizontal"] = df_horizontal["x_shift_diff"].mean() * RESOLUTION_DECS
        self.stats["horizontal_speed_std"] = df_horizontal["x_shift_diff"].std() * RESOLUTION_DECS

        # mask_vertical = pd.Series(False, index=df.index)
        # for start, end in self.getInvertedIntervals():
        #     mask_vertical |= (df['frame_ID'] >= start+5) & (df['frame_ID'] <= end-5)
        # df_vertical = df[mask_vertical]
        # self.speeds["vertical"] = df_vertical["y_shift_diff"].mean() * self.resolution_decs

        vertical_shifts = []
        for start, end in self.get_inverted_intervals():
            vertical_shift = df.loc[end, "y_shift"] - df.loc[start, "y_shift"]
            vertical_shifts.append(vertical_shift)
        self.speeds["vertical_shift"] = np.mean(vertical_shifts) * RESOLUTION_DECS
        self.stats["vertical_shift_std"] = np.std(vertical_shifts) * RESOLUTION_DECS

        print(
            f"Horizontal speed: {self.speeds['horizontal']}±{self.stats['horizontal_speed_std']}\nVertical shift: {self.speeds['vertical_shift']}±{self.stats['vertical_shift_std']}\nClockwise: {self.get_direction()}, Portrait: {self.is_portrait()}\nCalculated\n")

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
        return "CCW" if (self.speeds['horizontal'] > 0 and not self.is_portrait()) or (
                self.speeds['horizontal'] < 0 and self.is_portrait()) else "CW"

    def is_portrait(self):
        """
        Checks if the video has a portrait orientation.

        Returns:
            bool: True if the video is portrait, False otherwise.
        """
        # Uncomment for automatic detection based on video dimensions
        # return self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH) < self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return False

    def get_inverted_intervals(self):
        """
        Computes intervals that are inverted.

        Returns:
            np.ndarray: Array of inverted intervals.
        """
        return np.array([[self.intervals[i - 1][1], self.intervals[i][0]] for i in range(1, len(self.intervals))])

    def compute_frames_per360(self):
        """
        Calculates the number of frames required for a 360-degree rotation.
        """
        results = []

        frame_shift = int(np.ceil(np.mean(self.intervals[:, 1] - self.intervals[:, 0]) / ROW_OVERLAP))

        for start, end in self.intervals:
            frame = int(start + 50)
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            stat_a, a = self.video_capture.read()
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame + frame_shift)
            stat_b, b = self.video_capture.read()

            if not stat_a:
                print(f"Unable to read frame: {frame}")
            elif not stat_b:
                print(f"Unable to read frame: {frame + frame_shift}")
            else:

                feature_params = dict(maxCorners=100,
                                      qualityLevel=0.1,
                                      minDistance=7,
                                      blockSize=7)

                lk_params = dict(winSize=(15, 15),
                                 maxLevel=2,
                                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

                err_threshold = 9

                a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
                b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

                corners = cv2.goodFeaturesToTrack(a_gray, **feature_params)

                p1, st, err = cv2.calcOpticalFlowPyrLK(a_gray, b_gray, corners, None, **lk_params)

                # st = (st == 1) & (err < err_threshold)
                p1 = p1[st == 1]
                p0 = corners[st == 1]

                move = np.mean(p1 - p0, axis=0)

                results.append(move[0])

        self.frames_per_360 = np.ceil(frame_shift + np.mean(results) / abs(self.speeds['horizontal'])).astype(int)
        print(f"Frames per 360: {self.frames_per_360} calculated from frame_shift: {frame_shift}")

    def get_frames_per360(self):
        """
        Retrieves the number of frames required for a 360-degree rotation.

        Returns:
            int: Frames per 360-degree rotation.
        """
        return self.frames_per_360
