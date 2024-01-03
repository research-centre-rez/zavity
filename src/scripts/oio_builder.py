import os

import cv2
import imageio.v3 as iio
import numpy as np

import src.pipelines.oio_building_pipeline as oio_builder
from src.config import SRC, FRAMES_PER_360_DEG
from src.steps.video_frame_lightness import VideoLightness

if __name__ == "__main__":
    video_file_path = os.path.join(SRC, "GX010968.MP4")
    lightness = VideoLightness(video_file_path)
    lightness.process()
    mn, mx = lightness.rows_frame_no_start_end()[0]

    vidcap = cv2.VideoCapture(video_file_path)

    start = mn + (mx - mn) // 2 - FRAMES_PER_360_DEG // 2
    end = start + FRAMES_PER_360_DEG

    row = oio_builder.construct_row(vidcap, start, end)
    iio.imwrite(video_file_path.replace(".MP4", f"-oio-{start}-{end}.png"), row.astype(np.uint8))
