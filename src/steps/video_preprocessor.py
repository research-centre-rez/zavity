import os
import subprocess
import tempfile
from typing import LiteralString

import cv2
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from tqdm.auto import tqdm

from config import (Y1, Y2, X1, X2, SAMPLES, EVERY_NTH_FRAME, ROT_PER_FRAME, SIGMA, PADDING, CODEC, EXT, BORDER_BPS,
                    REFINED_BP, RECTIFY, INTERVAL_FILTER_TH, DEVERNAY_DOWNSCALE, SEGMENT_TYPE_THRESHOLD)


class VideoPreprocessor:
    """
    A class for preprocessing videos by analyzing frames, computing angles, and
    applying rotation corrections. Outputs a processed video and related computed data.

    Attributes:
        video_name (str): Name of the input video file.
        output_video_file_path (str): Path to the output processed video file.
        video_capture (cv2.VideoCapture): Video capture object for reading frames.
        angles (list): List of computed angles for each frame.
        borderBreakpoints (list): List of border breakpoints derived from the video.
        calc_rot_per_frame (bool): Flag indicating whether to calculate rotation per frame.
        rotation_per_frame (float): Calculated rotation per frame.
        breakpoints (np.ndarray): Detected breakpoints in the video based on angles.
        segment_type (np.ndarray): Segment type array representing trends in angles.
    """
    video_name: str
    output_video_file_path: LiteralString | str | bytes
    video_capture: cv2.VideoCapture
    angles: list
    borderBreakpoints: list
    calc_rot_per_frame: bool
    rotation_per_frame: float
    breakpoints: np.ndarray
    segment_type: np.ndarray

    def __init__(self, video_path, output_path, calc_rot_per_frame):
        """
        Initializes the VideoPreprocessor.

        Args:
            video_path (str): Path to the input video file.
            output_path (str): Directory where output files will be saved.
            calc_rot_per_frame (bool): Whether to calculate rotation per frame.
        """
        self.video_name = os.path.basename(video_path)
        self.output_path = output_path
        self.output_video_file_path = os.path.join(output_path,
                                                   os.path.splitext(self.video_name)[0] + '_preprocessed' + EXT)
        self.video_capture = cv2.VideoCapture(video_path)
        self.angles = []
        self.borderBreakpoints = []
        self.calc_rot_per_frame = calc_rot_per_frame

    def process(self):
        """
        Processes the video, either loading precomputed data or performing computations
        for angles, breakpoints, and rotation corrections.
        """
        if os.path.isfile(self.get_output_video_file_path()) and os.path.isfile(self._dump_path("borderBreakpoints")):
            self.borderBreakpoints = np.load(self._dump_path("borderBreakpoints"))
            print(f"Loaded {self._dump_path('borderBreakpoints')}\n"
                  f"{self.borderBreakpoints}\n")
        else:
            print(f"Pre-processing video: {self.video_name}")
            self.load_or_compute()

    def _dump_path(self, object_name, extension='npy'):
        """
        Generates a path for saving or loading a specific object related to the video.

        Args:
            object_name (str): Name of the object to save/load.
            extension (str): File extension for the object. Defaults to 'npy'.

        Returns:
            str: Path to the file.
        """
        return os.path.join(self.output_path, os.path.splitext(self.video_name)[0] + f'-{object_name}.{extension}')

    def dump(self, name: str, obj):
        """
        Saves an object to a file using numpy's save function.

        Args:
            name (str): Name of the object.
            obj: The object to save.
        """
        np.save(self._dump_path(name), obj)

    def load_or_compute(self):
        """
        Loads or computes angles, breakpoints, and rotation data for the video.
        """
        if self.calc_rot_per_frame:
            if os.path.isfile(self._dump_path('full_angles')):
                angles = np.load(self._dump_path('full_angles'))
            else:
                angles = self.compute_angles(step=1)
                self.dump('full_angles', angles)
            breakpoints = self.compute_breakpoints(angles, filter_length=8, step=1, merge_threshold=10)
            self.plot_angles(angles, breakpoints)
            border_breakpoints = self.compute_border_breakpoints(breakpoints, 1)
            self.rotation_per_frame = self.compute_rotation_per_frame(angles, breakpoints, border_breakpoints)
            print(f"Calculated rotation per frame\n"
                  f"{self.rotation_per_frame}\n"
                  f"Precalculated rotation per frame\n"
                  f"{ROT_PER_FRAME}\n"
                  f"Difference: {self.rotation_per_frame - ROT_PER_FRAME}\n")
        else:
            self.rotation_per_frame = ROT_PER_FRAME
            print(f"Loaded precalculated rotation per frame\n"
                  f"{self.rotation_per_frame}\n")

        if BORDER_BPS and REFINED_BP:
            self.borderBreakpoints = BORDER_BPS
            print(f"Loaded Border Breakpoints from config\n"
                  f"{self.borderBreakpoints}\n")
        else:
            if os.path.isfile(self._dump_path("angles")):
                self.angles = np.load(self._dump_path("angles"))
                print(f"Loaded {self._dump_path('angles')}\n")
            else:
                self.angles = self.compute_angles()
                self.dump("angles", self.angles)

            if os.path.isfile(self._dump_path("breakpoints")) and os.path.isfile(self._dump_path("borderBreakpoints")):
                self.borderBreakpoints = np.load(self._dump_path("borderBreakpoints"))
                print(f"Loaded {self._dump_path('borderBreakpoints')}\n"
                      f"{self.borderBreakpoints}\n")
                self.breakpoints = np.load(self._dump_path("breakpoints"))
                print(f"Loaded {self._dump_path('breakpoints')}\n"
                      f"{self.breakpoints}\n")
            else:
                self.breakpoints = self.compute_breakpoints(self.angles, step=EVERY_NTH_FRAME)
                self.dump("breakpoints", self.breakpoints)
                border_breakpoints = self.compute_border_breakpoints(self.breakpoints, EVERY_NTH_FRAME)
                self.borderBreakpoints = self.refine_border_breakpoints(border_breakpoints, step=EVERY_NTH_FRAME)
                self.dump("borderBreakpoints", self.borderBreakpoints)

        self.compute()

    def compute(self):
        """
        Executes computation tasks such as plotting angles and preprocessing the video.
        """
        self.plot_angles(self.angles, border_breakpoints=self.borderBreakpoints, breakpoints=self.breakpoints,
                         step=EVERY_NTH_FRAME)
        self.preprocess_video()

    @staticmethod
    def unix_path_to_win_path(path):
        """
        Converts a Unix-style path to a Windows-style path.

        Args:
            path (str): Unix-style path.

        Returns:
            str: Windows-style path.
        """
        return path.replace("/mnt/c/", "C:/").replace("/", "\\\\")

    @staticmethod
    def win_path_to_unix_path(path):
        """
        Converts a Windows-style path to a Unix-style path.

        Args:
            path (str): Windows-style path.

        Returns:
            str: Unix-style path.
        """
        return path.replace("\\", "/").replace("C:/", "/mnt/c/")

    def compute_angles(self, start=0, end=None, step=EVERY_NTH_FRAME):
        """
        Computes angles for the video frames between start and end with a given step.

        Args:
            start (int): Starting frame index. Defaults to 0.
            end (int): Ending frame index. Defaults to None (end of video).
            step (int): Step size for frame iteration.

        Returns:
            list: Computed angles for each frame.
        """
        computed_angles = []
        if end is None:
            end = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in tqdm(range(start, end, step), desc=f"Computing histograms from {start} to {end} with step {step}"):
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, i)
            success, frame = self.video_capture.read()
            cropped_frame = frame[Y1:Y2, X1:X2, 0]
            cropped_frame = cv2.resize(cropped_frame, (cropped_frame.shape[0] // DEVERNAY_DOWNSCALE, cropped_frame.shape[1] // DEVERNAY_DOWNSCALE))
            otsu_threshold, _ = cv2.threshold(cropped_frame, 0, 255, cv2.THRESH_OTSU)

            with tempfile.NamedTemporaryFile(suffix=".pgm", delete=False) as tmpfile:
                filename = tmpfile.name
                iio.imwrite(tmpfile.name, cropped_frame)  # this must be a grayscale image

                output_path = tmpfile.name.replace(".pgm", ".txt")

                process = subprocess.Popen(
                    ["devernay", self.win_path_to_unix_path(tmpfile.name),
                     "-t", self.win_path_to_unix_path(output_path),
                     "-l", f"{otsu_threshold / 15}",
                     "-h", f"{otsu_threshold / 3}",
                     # "-p", f"/mnt/c/Users/fathe/OneDrive/Documents/UK/MFF/Thesis/output/sample{i}.pdf",
                     "-s", f"1"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                tmpfile.close()

            # Wait for the process to complete and get stdout and stderr
            stdout, stderr = process.communicate()

            # Check for errors in stderr
            if stderr:
                print("Error running devernay:", stderr.decode("utf-8"))
            if os.path.exists(output_path):
                with open(output_path, 'r') as output_file:
                    result = output_file.read()
            else:
                print(f"Output file {output_path} does not exist.")

            lines = result.split("\n")
            dev = []
            for line in lines:
                if line != "":
                    x, y = line.split(' ')
                    dev.append((float(x), float(y)))
            dev = np.array(dev)
            if len(dev) == 0:
                print(f"Devernay did not find any angles for frame: {i}")
                computed_angles.append(45)
            else:
                choice = np.random.randint(0, len(dev), SAMPLES)
                xx0 = np.matmul(dev[choice, 0].reshape(-1, 1), np.ones((1, len(choice))))
                yy0 = np.matmul(dev[choice, 1].reshape(-1, 1), np.ones((1, len(choice))))
                xx1 = np.matmul(np.ones((len(choice), 1)), dev[choice, 0].reshape(1, -1))
                yy1 = np.matmul(np.ones((len(choice), 1)), dev[choice, 1].reshape(1, -1))
                valid = np.zeros_like(xx0, dtype=bool)
                valid[xx0 != xx1] = 1
                angles = np.zeros_like(xx0, np.float32)
                angles[xx0 == xx1] = np.pi / 2
                angles[valid] = np.arctan((yy0[valid] - yy1[valid]) / (xx0[valid] - xx1[valid])).reshape(-1)
                angles[np.eye(SAMPLES, dtype=bool)] = np.nan

                filtered = np.rad2deg(np.abs(angles[~np.isnan(angles)]))

                if filename is not None and os.path.exists(filename):
                    os.remove(filename)

                angle = self.compute_angle(filtered)
                computed_angles.append(angle)

        print(f"Angles calculated from {start} to {end} with step {step}\n")
        return computed_angles

    @staticmethod
    def compute_angle(hist):
        """
        Computes the most frequent angle from a histogram of angles.

        Args:
            hist (list): Histogram of angles.

        Returns:
            float: Most frequent angle.
        """
        counts, values = np.histogram(hist, bins=SAMPLES)
        counts = gaussian_filter1d(counts, sigma=SIGMA)
        max_index = np.argmax(counts)

        return values[max_index]

    def compute_breakpoints(self, angles=None, filter_length=-1, step=EVERY_NTH_FRAME, threshold=SEGMENT_TYPE_THRESHOLD,
                            merge_threshold=EVERY_NTH_FRAME, secondary=True, segment_type_return=False):
        """
        Computes breakpoints in the angle data.

        Args:
            angles (list): List of angle values. Defaults to self.angles.
            filter_length (int): Filter size for smoothing the angles. Defaults to -1 (no filter).
            step (int): Step size for computation.
            threshold (float): Threshold for detecting changes in segments.
            merge_threshold (int): Threshold for merging close breakpoints.
            secondary (bool): Whether to apply secondary refinement.
            segment_type_return (bool): Whether to return segment types along with breakpoints.

        Returns:
            np.ndarray: Array of breakpoints.
        """
        if angles is None:
            angles = self.angles

        if filter_length >= 2:
            angles = savgol_filter(angles, window_length=filter_length, polyorder=1)

        derivative = np.diff(angles)
        threshold = threshold * step

        segment_type = np.zeros_like(derivative)
        segment_type[derivative > threshold] = 1  # Increasing
        segment_type[derivative < -threshold] = -1  # Decreasing

        breakpoints = np.where(np.diff(segment_type) != 0)[0] + 1

        breakpoints = breakpoints * step

        if len(breakpoints) > 1:
            breakpoints = self.merge_breakpoints(breakpoints, step, threshold=merge_threshold, secondary=secondary,
                                                 segment_type=segment_type)

        breakpoints = np.concatenate([[0], breakpoints, [(len(angles) - 1) * step]])

        print(f"Calculated: Breakpoints\n"
              f"{breakpoints}\n")

        if segment_type_return:
            return breakpoints, segment_type
        else:
            self.segment_type = segment_type
            return breakpoints

    @staticmethod
    def merge_breakpoints(breakpoints, step, threshold, segment_type, secondary=True):
        """
        Merges close breakpoints based on a threshold.

        Args:
            breakpoints (np.ndarray): List of breakpoints.
            step (int): Step size for computation.
            threshold (int): Threshold for merging close breakpoints.
            segment_type (np.ndarray): Segment type of the angles.
            secondary (bool): Whether to apply secondary refinement.

        Returns:
            np.ndarray: Array of merged breakpoints.
        """
        merged_breakpoints = []

        # Temporary group for close breakpoints
        current_group = [breakpoints[0]]

        # Iterate over the breakpoints
        for i in range(1, len(breakpoints)):
            # If the difference between consecutive breakpoints is below the threshold, group them
            if breakpoints[i] - breakpoints[i - 1] <= threshold:
                current_group.append(breakpoints[i])
            else:
                # If a current group is finished, calculate the rounded average and store it
                avg_breakpoint = int(round(np.mean(current_group)))
                merged_breakpoints.append(avg_breakpoint)
                # Start a new group
                current_group = [breakpoints[i]]

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

    def compute_border_breakpoints(self, breakpoints, step):
        """
        Computes border breakpoints between segments with different trends.

        Args:
            breakpoints (ndarray): List of breakpoints.
            step (int): Step size for computation.

        Returns:
            np.ndarray: Array of border breakpoints.
        """
        border_breakpoints = []
        for i in range(0, len(breakpoints) - 1):
            if stats.mode(self.segment_type[breakpoints[i] // step:breakpoints[i + 1] // step])[0] == 0:
                border_breakpoints.append([breakpoints[i], breakpoints[i + 1]])

        # borderBreakpoints.append([breakpoints[-1], ])

        print(f"Calculated: Border Breakpoints\n"
              f"{np.asarray(border_breakpoints)}\n")

        return np.asarray(border_breakpoints)

    def refine_border_breakpoints(self, border_breakpoints, step):
        """
        Refines the detected border breakpoints for greater accuracy.

        Args:
            border_breakpoints (ndarray): List of border breakpoints.
            step (int): Step size for computation.

        Returns:
            list: Refined list of border breakpoints.
        """
        refined = []

        for start, end in border_breakpoints:
            if start != 0:
                start_breakpoint = self.refine_breakpoint(start)
            else:
                start_breakpoint = 0

            if end + step < int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)):
                end_breakpoint = self.refine_breakpoint(end)
            else:
                end_breakpoint = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

            refined.append([start_breakpoint, end_breakpoint])

        print(f"Calculated: Refined Border Breakpoints\n"
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
        angles = self.compute_angles(bp - EVERY_NTH_FRAME, bp + EVERY_NTH_FRAME, 1)
        breakpoints, segment_type = self.compute_breakpoints(angles, step=1, merge_threshold=1,
                                                             segment_type_return=True)
        if len(breakpoints) == 3:
            refined_bp = breakpoints[1]
        elif len(breakpoints) > 3:
            try:
                refined_bp = self.get_zero_segment(breakpoints, segment_type)
            except Exception as e:
                print(f"Error in refinement: {e}")
                return bp
        else:
            print(f"No breakpoints found: {breakpoints}")
            return bp
        refined_bp = bp - EVERY_NTH_FRAME + refined_bp
        print(f"Breakpoint refinement: {bp} -> {refined_bp}")
        return int(bp - EVERY_NTH_FRAME + refined_bp)

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

    def compute_rotation_per_frame(self, angles, breakpoints, border_breakpoints):
        """
        Computes the rotation per frame based on angles, breakpoints, and border breakpoints.

        Args:
            angles (list): List of angle values.
            breakpoints (np.ndarray): Detected breakpoints.
            border_breakpoints (np.ndarray): Border breakpoints.

        Returns:
            float: Rotation per frame.
        """
        fa = []
        j = 0
        k = 0
        start, end = border_breakpoints[j]
        for i in range(0, len(breakpoints) - 1):
            if not start == breakpoints[i]:
                segment = stats.mode(self.segment_type[breakpoints[i]:breakpoints[i + 1]])[0]
                offset = -(k * 180)
                if segment == 1 and not start == breakpoints[i + 1]:
                    k += 1
                f = angles[breakpoints[i]:breakpoints[i + 1]] * -segment + offset
                fa = np.concatenate([fa, f])
            else:
                j += 1
                if j < len(border_breakpoints):
                    start, end = border_breakpoints[j]

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

        print(f"Calculated : RotationPerFrame\n"
              f"{-a}")

        return -a

    def preprocess_video(self):
        """
        Applies preprocessing to the video, including cropping, rotating frames,
        and saving the processed output.
        """
        i_row = 0
        if len(self.borderBreakpoints) >= 2:
            start, end = self.borderBreakpoints[i_row]
        else:
            start, end = 0, 999999

        if REFINED_BP:
            angle_breakpoint = REFINED_BP
            print(f"Loaded Refined Breakpoint from config\n"
                  f"{angle_breakpoint}\n")
        else:
            if self.angles[self.breakpoints[2] // EVERY_NTH_FRAME] < self.angles[
                self.breakpoints[1] // EVERY_NTH_FRAME]:
                angle_breakpoint = self.refine_breakpoint(self.breakpoints[2])
            else:
                angle_breakpoint = self.refine_breakpoint(self.breakpoints[3])

        angle = -self.rotation_per_frame * angle_breakpoint

        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        out = cv2.VideoWriter(str(self.get_output_video_file_path()),
                              apiPreference=cv2.CAP_FFMPEG,
                              fourcc=cv2.VideoWriter_fourcc(*CODEC),
                              fps=60.0,
                              frameSize=(X2 - X1, Y2 - Y1),
                              params=[
                                  cv2.VIDEOWRITER_PROP_DEPTH,
                                  cv2.CV_8U,
                                  cv2.VIDEOWRITER_PROP_IS_COLOR,
                                  0,  # false
                              ]
                              )

        for i in tqdm(range(total_frames), desc="PreProcessing frames"):
            success, frame = self.video_capture.read()
            if RECTIFY:
                h, w = frame.shape[:2]
                # print(os.listdir())
                mtx = np.load("src/mtx.npy")
                dist = np.load("src/dist.npy")
                new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
                frame = cv2.undistort(frame, mtx, dist, None, new_camera_mtx)
            frame = frame[Y1 - PADDING:Y2 + PADDING, X1 - PADDING:X2 + PADDING]
            rotate_matrix = cv2.getRotationMatrix2D((frame.shape[1] / 2, frame.shape[0] / 2), angle, 1)
            rotated_image = cv2.warpAffine(
                src=frame, M=rotate_matrix, dsize=(frame.shape[1], frame.shape[0]))[PADDING:Y2 - Y1 + PADDING,
                            PADDING:X2 - X1 + PADDING, 0]

            out.write(rotated_image.astype(np.uint8))

            if i < start:
                angle += self.rotation_per_frame
            if i == end:
                i_row += 1
                if i_row < len(self.borderBreakpoints):
                    start, end = self.borderBreakpoints[i_row]
                else:
                    print(f"Reached out of index {i_row}")

        out.release()

    def plot_angles(self, angles, breakpoints=None, border_breakpoints=None, step=1):
        """
        Plots angles and optionally marks breakpoints and border breakpoints.

        Args:
            angles (list): List of angles to plot.
            breakpoints (list, optional): List of breakpoints to mark on the plot.
            border_breakpoints (list, optional): List of border breakpoints to mark.
            step (int, optional): Step size for angle computation. Defaults to 1.
        """
        plt.figure(figsize=(15, 3))
        x = range(len(angles))
        x = [i * step for i in x]
        plt.plot(x, angles)
        if breakpoints is not None:
            for bp in breakpoints:
                plt.axvline(bp, color="blue")
        if border_breakpoints is not None:
            for start, end in self.borderBreakpoints:
                plt.axvline(start, color="green")
                plt.axvline(end, color="red")
        plt.show()
        plt.savefig(self._dump_path("angles", "png"))
        plt.close()

    def get_output_video_file_path(self):
        """
        Retrieves the path for the output processed video file.

        Returns:
            str: Path to the output video file.
        """
        return self.output_video_file_path

    def get_intervals(self):
        """
        Computes intervals between border breakpoints, filtering out small intervals.

        Returns:
            list: List of filtered intervals.
        """
        start = 0
        end = self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
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
