import os
import subprocess
import tempfile

import cv2
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from tqdm.auto import tqdm

from config import (Y1, Y2, X1, X2, SAMPLES, THRESHOLD_DISTANCE_FOR_BREAKPOINT_MERGE, EVERY_NTH_FRAME,
                    ROT_PER_FRAME, SIGMA, PADDING)


class VideoPreprocessor:
    video_name: str
    output_video_file_path: str
    video_capture: cv2.VideoCapture
    angles: list
    borderBreakpoints: list
    calc_rot_per_frame: bool
    rotation_per_frame: float
    breakpoints: np.ndarray
    segment_type: np.ndarray

    def __init__(self, video_path, output_path, calc_rot_per_frame):
        self.video_name = os.path.basename(video_path)
        self.output_path = output_path
        self.output_video_file_path = os.path.join(output_path,
                                                   os.path.splitext(self.video_name)[0] + '_preprocessed.mp4')
        self.video_capture = cv2.VideoCapture(video_path)
        self.angles = []
        self.borderBreakpoints = []
        self.calc_rot_per_frame = calc_rot_per_frame

    def process(self):
        if os.path.isfile(self.getOutputVideoFilePath()):
            pass
        else:
            print(f"Pre-processing video: {self.video_name}")
            self.load_or_compute()

    def _dump_path(self, object_name):
        return os.path.join(self.output_path, os.path.splitext(self.video_name)[0] + f'-{object_name}.npy')

    def dump(self, name: str, object):
        np.save(self._dump_path(name), object)

    def load_or_compute(self):
        if self.calc_rot_per_frame:
            if os.path.isfile(self._dump_path('full_angles')):
                angles = np.load(self._dump_path('full_angles'))
            else:
                angles = self.computeAngles(step=1)
                self.dump('full_angles', angles)
            breakpoints = self.computeBreakpoints(angles, filter=8, step=1, merge_threshold=5)
            self.plotAngles(angles, breakpoints)
            borderBreakpoints = self.computeBorderBreakpoints(breakpoints, 1)
            self.rotation_per_frame = self.computeRotationPerFrame(angles, breakpoints, borderBreakpoints)
            print(f"Calculated rotation per frame\n"
                  f"{self.rotation_per_frame}\n"
                  f"Precalculated rotation per frame\n"
                  f"{ROT_PER_FRAME}\n"
                  f"Difference: {self.rotation_per_frame - ROT_PER_FRAME}\n")
        else:
            self.rotation_per_frame = ROT_PER_FRAME
            print(f"Loaded precalculated rotation per frame\n"
                  f"{self.rotation_per_frame}\n")

        if os.path.isfile(self._dump_path("angles")):
            self.angles = np.load(self._dump_path("angles"))
            print(f"Loaded {self._dump_path('angles')}\n")
        else:
            self.angles = self.computeAngles()
            self.dump("angles", self.angles)

        if os.path.isfile(self._dump_path("breakpoints")) and os.path.isfile(self._dump_path("borderBreakpoints")):
            self.borderBreakpoints = np.load(self._dump_path("borderBreakpoints"))
            print(f"Loaded {self._dump_path('borderBreakpoints')}\n"
                  f"{self.borderBreakpoints}\n")
            self.breakpoints = np.load(self._dump_path("breakpoints"))
            print(f"Loaded {self._dump_path('breakpoints')}\n"
                  f"{self.breakpoints}\n")
        else:
            self.breakpoints = self.computeBreakpoints(self.angles, step=EVERY_NTH_FRAME,
                                                       merge_threshold=EVERY_NTH_FRAME)
            self.dump("breakpoints", self.breakpoints)
            borderBreakpoints = self.computeBorderBreakpoints(self.breakpoints, EVERY_NTH_FRAME)
            self.borderBreakpoints = self.refineBorderBreakpoints(borderBreakpoints, step=EVERY_NTH_FRAME)
            self.dump("borderBreakpoints", self.borderBreakpoints)

        self.compute()

    def compute(self):
        self.plotAngles(self.angles, borderBreakpoints=self.borderBreakpoints, step=EVERY_NTH_FRAME)
        self.preprocessVideo()

    @staticmethod
    def unixPathToWinPath(path):
        return path.replace("/mnt/c/", "C:/").replace("/", "\\\\")

    @staticmethod
    def winPathToUnixPath(path):
        return path.replace("\\", "/").replace("C:/", "/mnt/c/")

    def computeAngles(self, start=0, end=None, step=EVERY_NTH_FRAME):
        computed_angles = []
        if end is None:
            end = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in tqdm(range(start, end, step), desc=f"Computing histograms from {start} to {end} with step {step}"):
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, i)
            success, frame = self.video_capture.read()
            undistorted = frame[Y1:Y2, X1:X2, 0]
            undistorted = cv2.resize(undistorted, (undistorted.shape[0] // 4, undistorted.shape[1] // 4))
            otsu_threshold, _ = cv2.threshold(undistorted, 0, 255, cv2.THRESH_OTSU)

            with tempfile.NamedTemporaryFile(suffix=".pgm", delete=False) as tmpfile:
                filename = tmpfile.name
                iio.imwrite(tmpfile.name, undistorted)  # this must be a grayscale image

                output_path = tmpfile.name.replace(".pgm", ".txt")

                process = subprocess.Popen(
                    ["devernay", self.winPathToUnixPath(tmpfile.name),
                     "-t", self.winPathToUnixPath(output_path),
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
                print("Something wrong happen")
            samples = 1000
            choice = np.random.randint(0, len(dev), samples)
            xx0 = np.matmul(dev[choice, 0].reshape(-1, 1), np.ones((1, len(choice))))
            yy0 = np.matmul(dev[choice, 1].reshape(-1, 1), np.ones((1, len(choice))))
            xx1 = np.matmul(np.ones((len(choice), 1)), dev[choice, 0].reshape(1, -1))
            yy1 = np.matmul(np.ones((len(choice), 1)), dev[choice, 1].reshape(1, -1))
            valid = np.zeros_like(xx0, dtype=bool)
            valid[xx0 != xx1] = 1
            angles = np.zeros_like(xx0, np.float32)
            angles[xx0 == xx1] = np.pi / 2
            angles[valid] = np.arctan((yy0[valid] - yy1[valid]) / (xx0[valid] - xx1[valid])).reshape(-1)
            angles[np.eye(samples, dtype=bool)] = np.nan

            filtered = np.rad2deg(np.abs(angles[~np.isnan(angles)]))

            if filename is not None and os.path.exists(filename):
                os.remove(filename)

            angle = self.computeAngle(filtered)
            computed_angles.append(angle)

        print(f"Angles calculated from {start} to {end} with step {step}\n")
        return computed_angles

    def computeAngle(self, hist):
        counts, values = np.histogram(hist, bins=SAMPLES)
        counts = gaussian_filter1d(counts, sigma=SIGMA)
        max_index = np.argmax(counts)

        return values[max_index]

    def computeBreakpoints(self, angles=None, filter=-1, step=EVERY_NTH_FRAME, threshold=0.04,
                           merge_threshold=THRESHOLD_DISTANCE_FOR_BREAKPOINT_MERGE, secondary=True,
                           segment_type_return=False):

        if self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT) < 1000:
            step = 1
            merge_threshold = 5

        if angles is None:
            angles = self.angles

        if filter >= 2:
            angles = savgol_filter(angles, window_length=filter, polyorder=1)

        derivative = np.diff(angles)
        threshold = threshold * step

        segment_type = np.zeros_like(derivative)
        segment_type[derivative > threshold] = 1  # Increasing
        segment_type[derivative < -threshold] = -1  # Decreasing
        self.segment_type = segment_type

        breakpoints = np.where(np.diff(segment_type) != 0)[0] + 1

        breakpoints = breakpoints * step

        if len(breakpoints) > 1:
            breakpoints = self.mergeBreakpoints(breakpoints, step, threshold=merge_threshold, secondary=secondary)

        breakpoints = np.concatenate([[0], breakpoints, [(len(angles) - 1) * step]])

        print(f"Calculated: Breakpoints\n"
              f"{breakpoints}\n")

        if segment_type_return:
            return breakpoints, segment_type
        else:
            return breakpoints

    def mergeBreakpoints(self, breakpoints, step, threshold=THRESHOLD_DISTANCE_FOR_BREAKPOINT_MERGE, secondary=True):

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
            if stats.mode(self.segment_type[:merged_breakpoints[0] // step])[0] == \
                    stats.mode(self.segment_type[merged_breakpoints[0] // step:merged_breakpoints[1] // step])[0]:
                new_merged_breakpoints = []
            else:
                new_merged_breakpoints = [merged_breakpoints[0]]

            for i in range(1, len(merged_breakpoints) - 1):
                # print(i, stats.mode(self.segment_type[merged_breakpoints[i - 1]:merged_breakpoints[i]])[0], stats.mode(self.segment_type[merged_breakpoints[i]:merged_breakpoints[i + 1]])[0])
                if not stats.mode(self.segment_type[merged_breakpoints[i - 1] // step:merged_breakpoints[i] // step])[
                           0] == \
                       stats.mode(self.segment_type[merged_breakpoints[i] // step:merged_breakpoints[i + 1] // step])[
                           0]:
                    new_merged_breakpoints.append(merged_breakpoints[i])

            if not stats.mode(self.segment_type[merged_breakpoints[-1] // step:])[0] == \
                   stats.mode(self.segment_type[merged_breakpoints[-2] // step:merged_breakpoints[-1] // step])[0]:
                new_merged_breakpoints.append(merged_breakpoints[-1])

            return np.asarray(new_merged_breakpoints)
        else:
            return np.asarray(merged_breakpoints)

    def computeBorderBreakpoints(self, breakpoints, step):
        borderBreakpoints = []
        for i in range(0, len(breakpoints) - 1):
            if stats.mode(self.segment_type[breakpoints[i] // step:breakpoints[i + 1] // step])[0] == 0:
                borderBreakpoints.append([breakpoints[i], breakpoints[i + 1]])

        # borderBreakpoints.append([breakpoints[-1], ])

        print(f"Calculated: Border Breakpoints\n"
              f"{np.asarray(borderBreakpoints)}\n")

        return np.asarray(borderBreakpoints)

    def refineBorderBreakpoints(self, borderBreakpoints, step):
        refined = []

        for start, end in borderBreakpoints:
            if start != 0:
                start_breakpoint = self.refineBreakpoint(start)
            else:
                start_breakpoint = 0

            if end + step < int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)):
                end_breakpoint = self.refineBreakpoint(end)
            else:
                end_breakpoint = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

            refined.append([start_breakpoint, end_breakpoint])

        print(f"Calculated: Refined Breakpoints\n"
              f"{refined}\n")

        return refined

    def refineBreakpoint(self, bp):
        angles = self.computeAngles(bp - EVERY_NTH_FRAME, bp + EVERY_NTH_FRAME, 1)
        breakpoints = self.computeBreakpoints(angles, step=1, filter=8, merge_threshold=3, secondary=True)
        if len(breakpoints) > 3:
            refined_bp = self.getZeroSegment(breakpoints)
        elif len(breakpoints) == 3:
            refined_bp = breakpoints[1]
        else:
            print(f"Too many breakpoints{breakpoints}")
            return bp
        return int(bp - EVERY_NTH_FRAME + refined_bp)

    def getZeroSegment(self, breakpoints):
        count_1 = np.count_nonzero(self.segment_type[breakpoints[0]:breakpoints[1]] == 0) / (
                breakpoints[1] - breakpoints[0])
        count_2 = np.count_nonzero(self.segment_type[breakpoints[-2]:breakpoints[-1]] == 0) / (
                breakpoints[-1] - breakpoints[-2])
        if count_1 > count_2:
            return breakpoints[1]
        else:
            return breakpoints[-2]

    def computeRotationPerFrame(self, angles, breakpoints, borderBreakpoints):
        fa = []

        j = 0
        if angles[breakpoints[2]] < angles[breakpoints[3]]:
            k = 0
        else:
            k = 1
        start, end = borderBreakpoints[j]
        for i in range(0, len(breakpoints) - 1):
            if not start == breakpoints[i]:
                segment = -stats.mode(self.segment_type[breakpoints[i]:breakpoints[i + 1]])[0]
                offset = -(k // 2 * 180)
                if not start == breakpoints[i + 1]:
                    k += 1
                f = angles[breakpoints[i]:breakpoints[i + 1]] * segment + offset
                fa = np.concatenate([fa, f])
            else:
                j += 1
                if j < len(borderBreakpoints):
                    start, end = borderBreakpoints[j]

        a, b = np.polyfit(np.arange(len(fa)), fa, 1)

        plt.plot(fa)
        plt.plot([(a * x + b) for x in np.arange(len(fa))])
        plt.show()

        print(f"Calculated : RotationPerFrame\n"
              f"{-a}")

        return -a

    def preprocessVideo(self):
        if len(self.borderBreakpoints) >= 2:
            i_row = 1
            start, end = self.borderBreakpoints[i_row]
            _, video_start = self.borderBreakpoints[0]
            video_end, _ = self.borderBreakpoints[-1]
        else:
            start = 0
            end = 999999
            _, video_start = self.borderBreakpoints[0]
            video_end, _ = self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)

        if self.angles[self.breakpoints[2] // EVERY_NTH_FRAME] < self.angles[self.breakpoints[1] // EVERY_NTH_FRAME]:
            angle_breakpoint = self.refineBreakpoint(self.breakpoints[2])
        else:
            angle_breakpoint = self.refineBreakpoint(self.breakpoints[3])

        angle = -self.rotation_per_frame * (angle_breakpoint - video_start)

        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, video_start)
        total_frames = video_end - video_start
        # total_frames = 100

        out = cv2.VideoWriter(self.getOutputVideoFilePath(),
                              apiPreference=cv2.CAP_FFMPEG,
                              fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
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
            frame = frame[Y1 - PADDING:Y2 + PADDING, X1 - PADDING:X2 + PADDING]
            rotate_matrix = cv2.getRotationMatrix2D((frame.shape[1] / 2, frame.shape[0] / 2), angle, 1)
            rotated_image = cv2.warpAffine(
                src=frame, M=rotate_matrix, dsize=(frame.shape[1], frame.shape[0]))[PADDING:Y2 - Y1 + PADDING,
                            PADDING:X2 - X1 + PADDING, 0]

            out.write(rotated_image.astype(np.uint8))

            if not (start <= i + video_start < end):
                angle += self.rotation_per_frame
                if i + video_start == end:
                    i_row += 1
                    if i_row < len(self.borderBreakpoints):
                        start, end = self.borderBreakpoints[i_row]
                    else:
                        print(f"Reached out of index {i_row}")

        out.release()

    def plotAngles(self, angles, breakpoints=None, borderBreakpoints=None, step=1):
        plt.figure(figsize=(15, 3))
        x = range(len(angles))
        x = [i * step for i in x]
        plt.plot(x, angles)
        if breakpoints is not None:
            for bp in breakpoints:
                plt.axvline(bp, color="blue")
        if borderBreakpoints is not None:
            for start, end in self.borderBreakpoints:
                plt.axvline(start, color="green")
                plt.axvline(end, color="red")
        plt.show()

    def getOutputVideoFilePath(self):
        return self.output_video_file_path
