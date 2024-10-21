import cv2
import subprocess
import matplotlib.pyplot as plt
import tempfile
import imageio.v3 as iio
import numpy as np
import os
from tqdm.auto import tqdm
from scipy.signal import savgol_filter


class VideoPreprocessor:
    video_file_path: str
    video_file_paths: list[str]
    video_capture: cv2.VideoCapture
    angles: list
    breakpoints: list
    rotation_per_frame: float
    borderBreakpoints: list
    segment_type: np.ndarray
    output_video_file_path: str
    Y1 = 550
    Y2 = 1800
    X1 = 1400
    X2 = 2650
    SAMPLES = 1000
    THRESHOLD_DISTANCE_FOR_BREAKPOINT_MERGE = 10
    FRAME_RATE = 40

    def __init__(self, video, SRC):
        self.video = video
        self.video_file_path = str(os.path.join(SRC, video))
        self.video_capture = cv2.VideoCapture(self.video_file_path)
        self.angles = []
        self.breakpoints = []
        self.borderBreakpoints = []
        self.video_file_paths = []
        self.output_video_file_path = f'/Users/fathe/OneDrive/Documents/UK/MFF/Thesis/output/{self.video}_preprocessed.mp4'

    def process(self):
        self.load_or_compute()

    def _dump_path(self, object):
        return f"{self.video_file_path}-{object}.npy"

    def dump(self, name: str, object):
        np.save(self._dump_path(name), object)

    def load_or_compute(self):
        if os.path.isfile("/rotation_per_frame.npy"):
            self.rotation_per_frame = np.load("/rotation_per_frame.npy")
            print(f"Loaded /rotation_per_frame.npy /n"
                  f"value: {self.rotation_per_frame}")
        else:
            angles = self.computeAngles(step=1)
            self.dump("angles", angles)
            breakpoints = self.computeBreakpoints(angles, filter=11, step=1)
            self.rotation_per_frame = self.computeRotationPerFrame(angles, breakpoints)
            np.save("/rotation_per_frame.npy", self.rotation_per_frame)

        if os.path.isfile(self._dump_path("angles")):
            self.angles = np.load(self._dump_path("angles"))
            print(f"Loaded {self._dump_path('angles')}")
        else:
            self.angles = self.computeAngles()
            self.dump("angles", self.angles)
        if os.path.isfile(f'/Users/fathe/OneDrive/Documents/UK/MFF/Thesis/output/{self.video}.mp4'):
            pass
        else:
            self.compute()

    def compute(self):
        self.breakpoints = self.computeBreakpoints()
        self.computeBorderBreakpoints()
        self.plotAngles()
        self.preprocessVideo()

    @staticmethod
    def unixPathToWinPath(path):
        return path.replace("/mnt/c/", "C:/").replace("/", "\\\\")

    @staticmethod
    def winPathToUnixPath(path):
        return path.replace("\\", "/").replace("C:/", "/mnt/c/")

    def computeAngles(self, start=0, end=None, step=None):
        computed_angles = []
        if end is None:
            end = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if step is None:
            step = self.FRAME_RATE

        for i in tqdm(range(start, end, step), desc=f"Computing histograms from {start} to {end} with step {step}"):
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, i)
            success, frame = self.video_capture.read()
            undistorted = frame[self.Y1:self.Y2, self.X1:self.X2, 0]
            undistorted = cv2.resize(undistorted, (undistorted.shape[0] // 4, undistorted.shape[1] // 4))
            otsu_threshold, _ = cv2.threshold(undistorted, 0, 255, cv2.THRESH_OTSU)

            with tempfile.NamedTemporaryFile(suffix=".pgm", delete=False) as tmpfile:
                filename = tmpfile.name
                iio.imwrite(tmpfile.name, undistorted)  # this must be a grayscale image

                output_path = tmpfile.name.replace(".pgm", ".txt")


                process = subprocess.Popen(
                    ["wsl", "devernay", self.winPathToUnixPath(tmpfile.name),
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

        print(f"Angles calculated from {start} to {end} with step {step}")
        return computed_angles

    def computeAngle(self, hist):
        counts, values = np.histogram(hist, bins=self.SAMPLES)
        max_index = np.argmax(counts)

        if values[max_index] > 89.90:
            counts[max_index] = -1
            max_index = np.argmax(counts)
            if values[max_index] >= 89:
                angle = 89.91
            elif values[max_index] == 0:
                counts[max_index] = -1
                max_index = np.argmax(counts)
                if values[max_index] < 0.1:
                    angle = 0
                else:
                    angle = values[max_index]
            else:
                angle = values[max_index]
        elif values[max_index] == 0:
            counts[max_index] = -1
            max_index = np.argmax(counts)
            if values[max_index] < 0.1:
                angle = 0
            elif values[max_index] >= 89:
                counts[max_index] = -1
                max_index = np.argmax(counts)
                if values[max_index] >= 89.8:
                    angle = 89.91
                else:
                    angle = values[max_index]
            else:
                angle = values[max_index]
        else:
            angle = values[max_index]

        return angle

    def computeBreakpoints(self, angles=None, filter=-1, step=None):

        if angles is None:
            angles = self.angles

        if step is None:
            step = self.FRAME_RATE

        if filter >= 2:
            angles = savgol_filter(angles, window_length=filter, polyorder=1)

        derivative = np.diff(angles)
        threshold = 0.04*step

        segment_type = np.zeros_like(derivative)
        segment_type[derivative > threshold] = 1  # Increasing
        segment_type[derivative < -threshold] = -1  # Decreasing

        breakpoints = np.where(np.diff(segment_type) != 0)[0] + 1

        if step != self.FRAME_RATE:
            breakpoints = self.mergeBreakpoints(breakpoints, threshold=9999999)
        else:
            self.segment_type = segment_type

        print(f"Breakpoints: {breakpoints} Calculated")
        return breakpoints

    def mergeBreakpoints(self, breakpoints, threshold=None):

        if threshold is None:
            threshold = self.THRESHOLD_DISTANCE_FOR_BREAKPOINT_MERGE

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

        return merged_breakpoints

    def computeBorderBreakpoints(self):
        self.borderBreakpoints.append([0, self.breakpoints[0]])
        for i in range(0, len(self.breakpoints) - 1):
            if self.segment_type[(self.breakpoints[i] + self.breakpoints[i + 1]) // 2] == 0:
                self.borderBreakpoints.append([self.breakpoints[i], self.breakpoints[i+1]])

        print(f"Border Breakpoints: {self.borderBreakpoints} Calculated")

    # TODO rychlejsi alternativa: odhadnout dvema primkamy - iteracni metoda s odstranovanim outlieru
    def refineBorderBreakpoints(self):
        refined = []

        for start, end in self.borderBreakpoints:
            start_angles = self.computeAngles(start-self.FRAME_RATE/2, start+self.FRAME_RATE/2, 1)
            start_breakpoint = self.computeBreakpoints(start_angles, step=1)[0]

            end_angles = self.computeAngles(start - self.FRAME_RATE / 2, start + self.FRAME_RATE / 2, 1)
            end_breakpoint = self.computeBreakpoints(end_angles, step=1)[0]

            refined.append([start_breakpoint, end_breakpoint])

        print(f"Refined Breakpoints: {refined} Calculated\n")

    def computeRotationPerFrame(self, angles, breakpoints):
        fa = angles[breakpoints[0]:breakpoints[1]] * -self.segment_type[
            (breakpoints[0] + breakpoints[1]) // 2]
        for i in range(1, len(breakpoints) - 1):
            f = angles[breakpoints[i]:breakpoints[i + 1]] * -self.segment_type[
                (breakpoints[i] + breakpoints[i + 1]) // 2] - (i // 2 * 180)
            fa = np.concatenate([fa, f])

        a, b = np.polyfit(np.arange(len(fa)), fa, 1)

        plt.plot(fa)
        plt.plot([(a * x + b) for x in np.arange(len(fa))])
        plt.show()

        print(f"Calculated : RotationPerFrame/n"
              f"value: {self.rotation_per_frame}")

        return -a

    def preprocessVideo(self):
        angle = -self.rotation_per_frame * (self.breakpoints[1] - self.breakpoints[0])

        i_row = 0
        start, end = self.borderBreakpoints[i_row]

        total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        out = cv2.VideoWriter(self.output_video_file_path,
                              apiPreference=cv2.CAP_FFMPEG,
                              fourcc=cv2.VideoWriter_fourcc(*'h264'),
                              fps=20.0,
                              frameSize=(self.Y2 - self.Y1, self.X2 - self.X1),
                              params=[
                                  cv2.VIDEOWRITER_PROP_DEPTH,
                                  cv2.CV_8U,
                                  cv2.VIDEOWRITER_PROP_IS_COLOR,
                                  0,  # false
                              ]
                              )

        for i in tqdm(range(total_frames), desc="PreProcessing frames"):
            success, frame = self.video_capture.read()
            rotate_matrix = cv2.getRotationMatrix2D((frame.shape[1] / 2, frame.shape[0] / 2), angle, 1)
            rotated_image = cv2.warpAffine(
                src=frame, M=rotate_matrix, dsize=(frame.shape[1], frame.shape[0]))[self.Y1: self.Y2,
                            self.X1:self.X2, 0]

            out.write(rotated_image.astype(np.uint8))

            if not (start >= i > end):
                angle += self.rotation_per_frame
            elif i >= end:
                i_row += 1
                start, end = self.borderBreakpoints[i_row]

        out.release()

    def plotAngles(self):
        plt.figure(figsize=(15, 3))
        plt.plot(self.angles)
        for bp in self.breakpoints:
            plt.axvline(bp, color="blue")
        for start, end in self.borderBreakpoints:
            plt.axvline(start, color="green")
            plt.axvline(end, color="red")
        plt.show()

    def getOutputVideoFilePaths(self):
        return self.output_video_file_path
