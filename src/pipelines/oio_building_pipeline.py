import cv2
import numpy as np
from tqdm.auto import tqdm

from src.config import BLENDED_PIXELS_PER_FRAME, SHIFT_PER_FRAME, FRAMES_PER_360_DEG, FRAME_SIZE


def construct_row(vidcap,
                  start: int,
                  end: int,
                  direction: str = "CCW",
                  shift_per_frame=SHIFT_PER_FRAME,
                  blended_pixels_per_frame=BLENDED_PIXELS_PER_FRAME):
    if not vidcap or end - start <= 0:
        raise IOError("Unsupported input data.")

    # pixel intensity accumulator
    frame_shift_to_pixels_total = np.ceil(FRAMES_PER_360_DEG * shift_per_frame + blended_pixels_per_frame).astype(int)
    row_image = np.zeros(
        (FRAME_SIZE[0],
         frame_shift_to_pixels_total))
    # weight matrix for the accumulator
    weight_matrix = np.zeros(row_image.shape)

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)

    for frameNo in tqdm(range(0, end - start), desc="Building row image"):
        _, image = vidcap.read()
        shift = (row_image.shape[1] - shift_per_frame * frameNo - FRAME_SIZE[1] // 2) if direction == "CCW" else (
                shift_per_frame * frameNo)
        shift_partial = shift % 1
        shift_matrix = np.float32([
            [1, 0, shift_partial],
            [0, 1, 0]
        ])

        aligned_image = cv2.warpAffine(image,
                                       shift_matrix,
                                       (FRAME_SIZE[1] + 1, FRAME_SIZE[0]))

        # we take column from the middle of the frame
        crop_x_start = (np.round(row_image.shape[1] - shift_per_frame * frameNo - blended_pixels_per_frame).astype(int)
                        if direction == "CCW"
                        else np.round(
            shift_per_frame * frameNo + blended_pixels_per_frame).astype(int))
        crop_x_end = np.max([0, crop_x_start + blended_pixels_per_frame])
        row_image[:, crop_x_start:crop_x_end] += aligned_image[:,
                                                 image.shape[1] // 2 - blended_pixels_per_frame // 2: image.shape[
                                                                                                          1] // 2 + blended_pixels_per_frame // 2,
                                                 0]
        weight_matrix[:, crop_x_start:crop_x_end] += 1

    return np.copy((row_image / weight_matrix))
