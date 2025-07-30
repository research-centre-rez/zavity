import os
import multiprocessing
from tqdm.auto import tqdm
import logging
import numpy as np
import cv2
import pandas as pd
from scipy.optimize import minimize
from config.config import OPTICS_RADIUS_PX
from file_utils import dump_path, dump_csv, load_csv, file_exists
from scipy.signal import savgol_filter

CROPPED_FRAME_SIDE_PX = np.sqrt(2 * np.power(OPTICS_RADIUS_PX, 2)).astype(int)


class AdaptiveFrameCropper:
    radius: int # circle fitting radius (depends on object size in the frame)
    video_path: str
    video_name: str
    num_frames: int # number of frames in the video (could not be precise)
    file_dump_name: str

    def __init__(self, video_path, radius=1200):
        self.video_path = video_path
        self.video_name = os.path.basename(self.video_path)
        self.radius = radius
        vidcap = cv2.VideoCapture(video_path)
        self.num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.file_dump_name = "frameCenters"

    @staticmethod
    def frame_center_estimate(frame, object_size_radius, initial_guess=None, **cv2_good_features_to_track_params):
        def circle_error(center):
            return np.sum(
                np.sqrt(
                    np.power(features[:, 0, 0] - center[0], 2) + np.power(features[:, 0, 1] - center[1], 2)) > object_size_radius
            )

        maxCorners = cv2_good_features_to_track_params.get("maxCorners", 500)
        qualityLevel = cv2_good_features_to_track_params.get("qualityLevel", 0.01)
        minDistance = cv2_good_features_to_track_params.get("minDistance", 50)

        features = cv2.goodFeaturesToTrack(frame, maxCorners, qualityLevel, minDistance)
        circle_position = minimize(circle_error,
                                   x0=initial_guess if initial_guess is not None else np.array(frame.shape)[::-1] / 2,
                                   method="Nelder-Mead")
        return circle_position.x

    def _frame_centers_estimates(self, params):
        start, end = params
        vidcap = cv2.VideoCapture(self.video_path)
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)

        centers = []
        for frame_no in tqdm(np.arange(start, end), total=end-start, desc=f"Centers (chunk): {start}"):
            success, frame = vidcap.read()
            if not success:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.uint8)
            location = AdaptiveFrameCropper.frame_center_estimate(frame, self.radius, centers[-1][1] if len(centers) > 0 else None)
            centers.append((frame_no, location))

        return centers

    def get_frames_center(self, start=0, end=None):
        if file_exists(self.video_name, self.file_dump_name, extension="csv"):
            computed_centers = load_csv(self.video_name, self.file_dump_name)
            logging.debug(f"Loaded {dump_path(self.video_name, self.file_dump_name, extension="csv")}")
            return computed_centers

        if end is None:
            end = self.num_frames

        cpus = multiprocessing.cpu_count() - 1
        process_chunk_size = (end - start) // cpus
        with multiprocessing.Pool(cpus) as pool:
            results = list(tqdm(pool.imap(self._frame_centers_estimates,
                                          [(frame_no, frame_no + process_chunk_size)
                                           for frame_no in range(start, end, process_chunk_size)]),
                                total=cpus,  # (end-start)/step,
                                desc=f"Computing frame centers from {start} to {end} with step {process_chunk_size}")
                           )

        computed_centers = np.array(
            sorted([(frame_no, position[0], position[1]) for records in results for frame_no, position in records], key=lambda x: x[0]))

        # NOTE: maybe this should be done in 2D directly
        stabilized = np.stack([
            computed_centers[:, 0],
            savgol_filter(computed_centers[:, 1], 1500, 5),
            savgol_filter(computed_centers[:, 2], 1500, 5)
        ], axis=1)

        logging.debug(f"Centers calculated from {start} to {end} with chunk size {process_chunk_size}\n")
        dump_csv(self.video_name, self.file_dump_name, pd.DataFrame(stabilized, columns=["frame number", "cx", "cy"]))

        return stabilized

    @staticmethod
    def crop(frame, cx, cy):
        return frame[int(cy) - CROPPED_FRAME_SIDE_PX // 2: int(cy) + CROPPED_FRAME_SIDE_PX // 2,
               int(cx) - CROPPED_FRAME_SIDE_PX // 2: int(cx) + CROPPED_FRAME_SIDE_PX // 2]

if __name__ == "__main__":
    afc = AdaptiveFrameCropper("/Users/gimli/cvr/data/zavity/ETE 2025_07_01-sample/in/Hor_ZH2_down.MP4")
    centers = afc.get_frames_center()
    with open("/Users/gimli/temp/Hor_ZH2_down.csv", "wt") as f:
        f.write("frame number;x;y\n")
        for f_no, location in centers:
            f.write(f"{f_no};{location[0]};{location[1]}\n")
