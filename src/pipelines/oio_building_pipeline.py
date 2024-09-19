import cv2
import numpy as np
from tqdm.auto import tqdm

from src.config import BLENDED_PIXELS_PER_FRAME, SHIFT_PER_FRAME, FRAMES_PER_360_DEG, FRAME_SIZE, BLENDED_PIXELS_SHIFT


def construct_row(vidcap,
                  start: int,
                  end: int,
                  direction: str = "CCW",
                  shift_per_frame=SHIFT_PER_FRAME,
                  blended_pixels_per_frame=BLENDED_PIXELS_PER_FRAME,
                  blended_pixels_shift=BLENDED_PIXELS_SHIFT,
                  frames_per_360_deg=FRAMES_PER_360_DEG,
                  rotation: bool = False):
    if not vidcap or end - start <= 0:
        raise IOError("Unsupported input data.")

    # pixel intensity accumulator
    frame_shift_to_pixels_total = np.ceil(frames_per_360_deg * shift_per_frame + blended_pixels_per_frame).astype(int)
    # print(frame_shift_to_pixels_total, shift_per_frame)
    row_image = np.zeros(
        (FRAME_SIZE[0],
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
            shift = (row_image.shape[1] - shift_per_frame * frameNo - FRAME_SIZE[1] // 2) if direction == "CCW" else (
                    shift_per_frame * frameNo)
            shift_partial = shift % 1
            shift_matrix = np.float32([
                [1, 0, shift_partial],
                [0, 1, 0]
            ])

            aligned_image = cv2.warpAffine(image, shift_matrix, (FRAME_SIZE[1] + 1, FRAME_SIZE[0]))
        # except: print(status, image, vidcap.get(cv2.CAP_PROP_POS_FRAMES), vidcap.get(cv2.CAP_PROP_FRAME_COUNT),
        # shift, shift_partial)

        # we take column from the middle of the frame
        crop_x_start = (np.round(row_image.shape[1] - shift_per_frame * frameNo - blended_pixels_per_frame).astype(int)
                        if direction == "CCW"
                        else np.round(shift_per_frame * frameNo).astype(int))
        crop_x_end = np.max([0, crop_x_start + blended_pixels_per_frame])
        # try:
        row_image[:, crop_x_start:crop_x_end] += aligned_image[:, (image_part // 2 - blended_pixels_per_frame // 2) + blended_pixels_shift : (image_part // 2 + blended_pixels_per_frame // 2) + blended_pixels_shift, 0]
        # except:
        #     print(crop_x_start, crop_x_end, image.shape, aligned_image.shape)
        weight_matrix[:, crop_x_start:crop_x_end] += 1

    return np.copy((row_image / weight_matrix))
