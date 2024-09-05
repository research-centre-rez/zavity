import glob
import os

import cv2
import imageio.v3 as iio
import numpy as np

import src.pipelines.oio_building_pipeline as oio_builder
import src.steps.video_camera_motion as motion
from src.config import SRC, FRAMES_PER_360_DEG
from src.steps.video_frame_lightness import VideoLightness
from src.steps.video_camera_motion import VideoMotion
from steps.image_row_stitcher import phacorr, stitch, stitch_images, regsum, register_template

if __name__ == "__main__":
    video_file_path = os.path.join(SRC, "GX010968.MP4")

    # lightness = VideoLightness(video_file_path)
    # lightness.process()
    # row_intervals = lightness.rows_frame_no_start_end()
    #
    # print(row_intervals)
    # print(len(row_intervals))

    motion = VideoMotion(video_file_path, 1, 8)
    motion.process()

    # vidcap = cv2.VideoCapture(video_file_path)
    # for mn, mx in motion.intervals:
    #     start = mn + (mx - mn) // 2 - FRAMES_PER_360_DEG // 2
    #     end = start + FRAMES_PER_360_DEG
    #     row = oio_builder.construct_row(vidcap, start, end)
    #     iio.imwrite(video_file_path.replace(".MP4", f"-oio-{start}-{end}.png"), row.astype(np.uint8))

    imgs = []
    for mn, mx in motion.intervals:
        start = mn + (mx - mn) // 2 - FRAMES_PER_360_DEG // 2
        end = start + FRAMES_PER_360_DEG
        path = video_file_path.replace(".MP4", f"-oio-{start}-{end}.png")
        img = cv2.imread(path)
        imgs.append(img)

    # stitch_images(imgs)

    # Get image dimensions
    height, width, _ = imgs[1].shape

    # Calculate the upper half
    upper_half = imgs[1][:height // 2, :]

    # Calculate the new width after cutting 20% from both sides
    new_width = int(width * 0.6)

    # Calculate the starting and ending points for the width
    start_col = int(width * 0.2)
    end_col = start_col + new_width

    # Crop the image
    cropped_image = upper_half[:, start_col:end_col]

    scaling = 0.05

    # cv2.imshow("img0", cv2.resize(imgs[0], (int(imgs[0].shape[1]*scaling), int(imgs[0].shape[0]*scaling))))
    # cv2.imshow("img1cropped", cv2.resize(cropped_image, (int(cropped_image.shape[1]*scaling), int(cropped_image.shape[0]*scaling))))


    cv2.imshow("regsum", cv2.resize(regsum(imgs[0], cropped_image), (int(imgs[0].shape[1]*scaling), int(imgs[0].shape[0]*scaling))))
    # cv2.imshow("regsum", cv2.resize(register_template(imgs[0], cropped_image), (int(imgs[0].shape[1]*scaling), int(imgs[0].shape[0]*scaling))))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # print(f'The shift is = {phacorr(cropped_image, imgs[0])}[down, right]  ')
    # print(f'The shift is = {phacorr(imgs[0], cropped_image)}[down, right]  ')
    # print(f'The shift is = {phacorr(imgs[0], imgs[1])}[down, right]  ')
    # print(f'The shift is = {phacorr(imgs[1], imgs[2])}[down, right]  ')
    # print(f'The shift is = {phacorr(imgs[2], imgs[3])}[down, right]  ')
    # print(f'The shift is = {phacorr(imgs[3], imgs[4])}[down, right]  ')
    # print(f'The shift is = {phacorr(imgs[4], imgs[5])}[down, right]  ')
    # print(f'The shift is = {phacorr(imgs[5], imgs[6])}[down, right]  ')

    # if status == 0:
    #     iio.imwrite(video_file_path.replace(".MP4", "-final_image.png"), final_image)
    #     cv2.imshow("Final Image", final_image)
    #     cv2.waitKey(0)
    # else:
    #     print(status)

    row = oio_builder.construct_row(vidcap, start, end)
    iio.imwrite(video_file_path.replace(".MP4", f"-oio-{start}-{end}.png"), row.astype(np.uint8))
