import os

import cv2
import imageio.v3 as iio
import numpy as np
from tqdm.auto import tqdm

from config import BLENDED_PIXELS_PER_FRAME, BLENDED_PIXELS_SHIFT


def construct_rows(motions, video_file_path, output_path):
    vidcap = cv2.VideoCapture(video_file_path)
    rows = []
    for i, interval in enumerate(motions.getIntervals()):
        mn, mx = interval
        start = mn + (mx - mn) // 2 - motions.getFramesPer360() // 2
        end = start + motions.getFramesPer360()
        file_path = os.path.join(output_path, os.path.splitext(os.path.basename(video_file_path))[0] + f"-oio-{i}.png")
        if not os.path.isfile(file_path):
            row = construct_row(vidcap, int(start), int(end), direction=motions.getDirection(),
                                rotation=motions.isPortrait(), shift_per_frame=motions.getHorizontalSpeed(),
                                frames_per_360_deg=motions.getFramesPer360())
            iio.imwrite(file_path, row.astype(np.uint8))
        else:
            row = iio.imread(file_path)
        rows.append(row)

    return rows


def construct_row(vidcap,
                  start: int,
                  end: int,
                  shift_per_frame,
                  frames_per_360_deg,
                  direction: str,
                  rotation: bool = False,
                  blended_pixels_per_frame=BLENDED_PIXELS_PER_FRAME,
                  blended_pixels_shift=BLENDED_PIXELS_SHIFT):
    if not vidcap or end - start <= 0:
        raise IOError("Unsupported input data.")

    frame_size = (int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # pixel intensity accumulator
    frame_shift_to_pixels_total = np.ceil(frames_per_360_deg * shift_per_frame + blended_pixels_per_frame).astype(int)
    # print(frame_shift_to_pixels_total, shift_per_frame)
    row_image = np.zeros(
        (frame_size[0],
         frame_shift_to_pixels_total))
    # weight matrix for the accumulator
    weight_matrix = np.zeros(row_image.shape)

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)

    if rotation:
        image_part = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    else:
        image_part = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    for frameNo in tqdm(range(0, end - start), desc="Building row image"):
        status, image = vidcap.read()
        if status:
            if rotation:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            shift = (row_image.shape[1] - shift_per_frame * frameNo - frame_size[1] // 2) if direction == "CCW" else (
                    shift_per_frame * frameNo)
            shift_partial = shift % 1
            shift_matrix = np.float32([
                [1, 0, shift_partial],
                [0, 1, 0]
            ])

            aligned_image = cv2.warpAffine(image, shift_matrix, (frame_size[1] + 1, frame_size[0]))

        # we take column from the middle of the frame
        crop_x_start = (np.round(row_image.shape[1] - shift_per_frame * frameNo - blended_pixels_per_frame).astype(int)
                        if direction == "CCW"
                        else np.round(shift_per_frame * frameNo).astype(int))
        crop_x_end = np.max([0, crop_x_start + blended_pixels_per_frame])
        row_image[:, crop_x_start:crop_x_end] += aligned_image[:, (
                                                                              image_part // 2 - blended_pixels_per_frame // 2) + blended_pixels_shift: (
                                                                                                                                                                   image_part // 2 + blended_pixels_per_frame // 2) + blended_pixels_shift,
                                                 0]
        weight_matrix[:, crop_x_start:crop_x_end] += 1

    return np.copy((row_image / weight_matrix))
