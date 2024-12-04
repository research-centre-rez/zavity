import os.path
import pickle

import cv2
import numpy as np
import pandas as pd
from scipy.stats import stats
from tqdm.auto import tqdm

from config import FPS_REDUCTION, RESOLUTION_DECS, ROW_OVERLAP


class VideoMotion:
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

    def __init__(self, video_file_path: str, output_path: str):
        self.speeds = {}
        self.stats = {}
        self.motion_directions = []
        self.motion_positions = []
        self.intervals = np.empty((0, 2))
        self.video_capture = cv2.VideoCapture(video_file_path)
        self.width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH) / RESOLUTION_DECS)
        self.height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT) / RESOLUTION_DECS)
        self.video_file_path = video_file_path
        self.output_path = output_path
        self.video_name = os.path.basename(video_file_path)

    @staticmethod
    def getCorners(gray_frame, **feature_params):
        corners = cv2.goodFeaturesToTrack(gray_frame, **feature_params)
        return corners

    def _dump_path(self, object_name):
        return os.path.join(self.output_path, os.path.splitext(self.video_name)[0] + f'-{object_name}.npy')

    def dump(self, name: str, object):
        np.save(self._dump_path(name), object)

    def load_or_compute(self):
        if os.path.isfile(self._dump_path("motion_directions")) and os.path.isfile(self._dump_path("motion_positions")):
            self.motion_directions = np.load(self._dump_path("motion_directions"))
            self.motion_positions = np.load(self._dump_path("motion_positions"))
        else:
            self.computeMotion()
            self.dump("motion_directions", self.motion_directions)
            self.dump("motion_positions", self.motion_positions)
        if os.path.isfile(self._dump_path("intervals")) and os.path.isfile(
                self._dump_path("speeds")) and os.path.isfile(self._dump_path("frames_per_360")) and os.path.isfile(
            self._dump_path("stats")):
            self.intervals = np.load(self._dump_path("intervals"))
            print(f"Intervals: {self.intervals} Loaded")
            with open(self._dump_path("speeds"), 'rb') as fp:
                self.speeds = pickle.load(fp)
            with open(self._dump_path("stats"), 'rb') as fp:
                self.stats = pickle.load(fp)
            print(f"Horizontal speed: {self.speeds['horizontal']}±{self.stats['horizontal_speed_std']}\n"
                  f"Vertical shift: {self.speeds['vertical_shift']}±{self.stats['vertical_shift_std']}\n"
                  f"Clockwise: {self.getDirection()}, Portrait: {self.isPortrait()}\nLoaded\n")
            self.frames_per_360 = np.load(self._dump_path("frames_per_360"))
            print(f"Frames per 360: {self.frames_per_360} Loaded")
        else:
            self.compute()

    def compute(self):
        self.computeIntervals()
        self.computeSpeeds()
        self.computeFramesPer360()
        self.dump("intervals", self.intervals)
        with open(self._dump_path("speeds"), 'wb') as fp:
            pickle.dump(self.speeds, fp)
        with open(self._dump_path("stats"), 'wb') as fp:
            pickle.dump(self.stats, fp)
        self.dump("frames_per_360", self.frames_per_360)

    @staticmethod
    def getCommonDirection(motion_direction):
        return stats.mode(motion_direction)

    def computeIntervals(self):
        desired_movement = self.getCommonDirection(self.motion_directions)[0]
        start = 0

        for i in range(1, len(self.motion_directions)):
            if self.motion_directions[i] != self.motion_directions[start]:
                if i - start > 10 and self.motion_directions[start] == desired_movement:
                    self.intervals = np.append(self.intervals, [[start, i - 1]], axis=0)
                start = i

        if len(self.motion_directions) - 1 - start > 10 and self.motion_directions[start] == desired_movement:
            self.intervals = np.append(self.intervals, [[start, len(self.motion_directions) - 1]], axis=0)

        self.merge_intervals(120)
        print(f"Intervals: {self.intervals} Calculated")

    def merge_intervals(self, max_gap):
        merged_intervals = [self.intervals[0]]

        for i in range(1, len(self.intervals)):
            prev_end = merged_intervals[-1][1]
            curr_start, curr_end = self.intervals[i]

            if curr_start - prev_end - 1 <= max_gap:
                # Merge intervals
                merged_intervals[-1] = (merged_intervals[-1][0], curr_end)
            else:
                # Add a new interval
                merged_intervals.append(self.intervals[i])

        self.intervals = np.array(merged_intervals)

    def process(self):
        print(f"Processing VideoMotion for: {self.video_file_path}\n")
        self.load_or_compute()

    def computeMotion(self):
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
            corners = self.getCorners(old_gray, **feature_params)

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

    def showFrames(self, frame_numbers):

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

    def getIntervals(self):
        return self.intervals

    def computeSpeeds(self):
        if self.isPortrait():
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
        for start, end in self.getInvertedIntervals():
            vertical_shift = df.loc[end, "y_shift"] - df.loc[start, "y_shift"]
            vertical_shifts.append(vertical_shift)
        self.speeds["vertical_shift"] = np.mean(vertical_shifts) * RESOLUTION_DECS
        self.stats["vertical_shift_std"] = np.std(vertical_shifts) * RESOLUTION_DECS

        print(
            f"Horizontal speed: {self.speeds['horizontal']}±{self.stats['horizontal_speed_std']}\nVertical shift: {self.speeds['vertical_shift']}±{self.stats['vertical_shift_std']}\nClockwise: {self.getDirection()}, Portrait: {self.isPortrait()}\nCalculated\n")

    def getHorizontalSpeed(self):
        return abs(self.speeds['horizontal'])

    def getVerticalSpeed(self):
        return abs(self.speeds['vertical'])

    def getAverageVerticalShift(self):
        return abs(self.speeds['vertical_shift'])

    def isMovingDown(self):
        return self.speeds['vertical_shift'] < 0

    def getDirection(self):
        return "CCW" if (self.speeds['horizontal'] > 0 and not self.isPortrait()) or (
                self.speeds['horizontal'] < 0 and self.isPortrait()) else "CW"

    def isPortrait(self):
        # if self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH) < self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT):
        #     return True
        # else:
        #     return False
        return False

    def getInvertedIntervals(self):
        return np.array([[self.intervals[i - 1][1], self.intervals[i][0]] for i in range(1, len(self.intervals))])

    def computeFramesPer360(self):
        results = []
        frame_shift = int(np.ceil(np.mean(self.intervals[:, 1] - self.intervals[:, 0]) / ROW_OVERLAP))

        for start, end in self.intervals:
            frame = int(start + 100)
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

                # print(move, p1.shape, p0.shape)

                result = np.ceil(frame_shift + move[0] / abs(self.speeds['horizontal'])).astype(int)
                # print(result)

                results.append(result)

        self.frames_per_360 = np.mean(results)
        print(f"Frames per 360: {self.frames_per_360} calculated from frame_shift: {frame_shift}")

    def getFramesPer360(self):
        return self.frames_per_360
