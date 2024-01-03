import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from tqdm.auto import tqdm

import src.steps.video_timing as timing
from src.config import LIGHT_ON_OFF_FRAME_DISTANCE


class VideoLightness:
    video_file_path: str
    video_timing: list[tuple[int, float]]
    frame_lightness = np.array([])
    light_on = None
    light_off = None

    def __init__(self, video_file_path, video_timing=None):
        self.video_file_path = video_file_path
        if video_timing is None:
            self.video_timing = timing.load_or_extract(video_file_path)
        else:
            self.video_timing = video_timing

    def process(self):
        self.load_or_compute()
        self.dump()
        self.cuts()

    def _dump_path(self):
        return f"{self.video_file_path}-frame_lightness.npy"

    def load_or_compute(self):
        if os.path.isfile(self._dump_path()):
            self.frame_lightness = np.load(self._dump_path())
            return self.frame_lightness
        else:
            return self.compute()

    def compute(self):
        if len(self.frame_lightness) != 0:
            return self.frame_lightness

        lightness = []
        vidcap = cv2.VideoCapture(self.video_file_path)
        success, image = vidcap.read()
        if not success:
            raise IOError(f"Error reading video file {self.video_file_path} at first frame.")
        for frameNo, timestamp in tqdm(self.video_timing[1:],
                                       total=len(self.video_timing[1:]),
                                       desc=f"Frame lightness computation ({self.video_file_path})"):
            success, image = vidcap.read()
            if not success:
                raise IOError(
                    f"Error reading video file {self.video_file_path} at frame {frameNo} with timestamp {timestamp}")
            lightness.append(np.sum(image))
        self.frame_lightness = np.array(lightness)

        return self.frame_lightness

    def dump(self):
        np.save(self._dump_path(), self.frame_lightness)

    def plot(self):
        fig = plt.figure(figsize=(15, 5))
        plt.plot(self.frame_lightness)
        plt.ylabel("frame lightness")
        plt.xlabel("frame number")
        plt.title("Frame lightness - sync row cuts")
        plt.show()

        return fig

    def plot_on_off(self):
        plt.figure(figsize=(15, 5))
        plt.plot(self.frame_lightness, color="gray")
        for ons in self.light_on[0]:
            plt.axvline(ons, color="green")
        for offs in self.light_off[0]:
            plt.axvline(offs, color="red")
        plt.ylabel("frame lightness")
        plt.xlabel("frame number")
        plt.title("Lookup whether cut detection were successful")
        plt.show()

    def cuts(self, light_on_off_frame_distance=LIGHT_ON_OFF_FRAME_DISTANCE):
        self.light_on, self.light_off = [
            find_peaks(
                signal,
                distance=light_on_off_frame_distance,
                height=2 * (np.max(signal) - np.min(signal)) / 3 + np.min(signal))
            for signal in [self.frame_lightness, -self.frame_lightness]
        ]

    def rows_frame_no_start_end(self):
        """
        :return: list of tuples frame_no_start, frame_no_end corresponding to a scanned row
        """
        return list(zip(self.light_on[0], self.light_off[0][1:]))
