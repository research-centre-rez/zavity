import math
import os
import cv2
import imageio.v3 as iio
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from tqdm.auto import tqdm

from config.config import TESTING_MODE, BLENDED_PIXELS_PER_FRAME, BLENDED_PIXELS_SHIFT, SINUSOID_SAMPLING, IMAGE_REPEATS


def calculate_movements(images, moving_down):
    """
    Calculates cumulative movements between columns from cumulative intensities from stripes from row image.

    Args:
        images (list[np.ndarray]): List of grayscale image frames.
        moving_down (bool): Indicates the movement direction.

    Returns:
        list[np.ndarray]: List of cumulative movements for each image frame.
    """
    movementses = []
    images = images.copy()
    for i, image in enumerate(images):
        if moving_down and i == 0:
            image = image[image.shape[0] // 4:, :]
        elif not moving_down and i == len(images) - 1:
            image = image[image.shape[0] // 4:, :]

        image = np.hstack((image,) * IMAGE_REPEATS)

        # print(image.shape)
        stripe_width = 100
        movements = []

        for i in range(0, image.shape[1] - stripe_width - SINUSOID_SAMPLING, SINUSOID_SAMPLING):
            # Extract vertical stripes
            left_stripe = image[:, i:stripe_width + i]
            right_stripe = image[:, i + SINUSOID_SAMPLING:stripe_width + i + SINUSOID_SAMPLING]

            # Sum pixel intensities along the x-axis
            left_sum = np.sum(left_stripe, axis=1)
            right_sum = np.sum(right_stripe, axis=1)

            left_sum = gaussian_filter1d(left_sum, sigma=23)
            right_sum = gaussian_filter1d(right_sum, sigma=23)

            distance = 150
            # Detect peaks
            peaks_left, _ = find_peaks(left_sum, distance=distance)
            peaks_right, _ = find_peaks(right_sum, distance=distance)

            try:
                if peaks_left.shape[0] > peaks_right.shape[0]:
                    movement1 = np.mean(peaks_left[1:] - peaks_right)
                    movement2 = np.mean(peaks_left[:-1] - peaks_right)
                    if abs(movement1) > abs(movement2):
                        movement = movement2
                    else:
                        movement = movement1

                elif peaks_left.shape[0] < peaks_right.shape[0]:
                    movement1 = np.mean(peaks_left - peaks_right[:-1])
                    movement2 = np.mean(peaks_left - peaks_right[1:])
                    if abs(movement1) > abs(movement2):
                        movement = movement2
                    else:
                        movement = movement1
                else:
                    movement = np.mean(peaks_left - peaks_right)

                if abs(movement) > 5:
                    movement = movements[-1]
                movements.append(movement)
            except:
                movements.append(movements[-1])
                # # Plot the results
                # plt.figure(figsize=(10, 5))
                # plt.plot(left_sum, label="Left Stripe")
                # plt.plot(right_sum, label="Right Stripe")
                # plt.title("Summed Pixel Intensities")
                # plt.xlabel("Vertical Position (pixels)")
                # plt.ylabel("Summed Intensity")
                # plt.legend()
                # plt.savefig(os.path.join(output_path, f"{video_file_path}{i}_{peaks_left}_{peaks_right}" + ".png"))

        movementses.append(np.cumsum(movements))

    return movementses


def detrend_movements(movementses, output_path, video_file_path):
    """
    Removes linear trends from cumulative movements.

    Args:
        movementses (list[np.ndarray]): List of cumulative movements.
        output_path (str): Directory to save debug plots.
        video_file_path (str): Video file path for labeling outputs.

    Returns:
        list[np.ndarray]: List of detrended cumulative movements.
    """
    detrended_movementses = []
    for i, movements in enumerate(movementses):
        x = np.arange(len(movements))
        coefficients = np.polyfit(x, movements, deg=1)
        linear_trend = np.polyval(coefficients[-2:], x)
        detrended_movementses.append(movements - linear_trend)

        if TESTING_MODE:
            # Plot to visualize detrending
            plt.figure(figsize=(10, 5))
            plt.plot(x, movements, label="Original Data")
            plt.plot(x, linear_trend, label="Fitted Line (Trend)")
            plt.plot(x, movements - linear_trend, label="Detrended Data")
            plt.legend()
            plt.xlabel("Stripe Index")
            plt.ylabel("Movement")
            plt.title("Detrending the Cumulative Movement")
            plt.grid(True)
            plt.savefig(os.path.join(output_path, f"{os.path.splitext(video_file_path)[0]}_oio_{i}_detrend" + ".png"))

    return detrended_movementses

def sinusoid(x, A, B, C, D):
    """
    Defines a sinusoidal function.

    Args:
        x (np.ndarray | float): Input values.
        A (float): Amplitude.
        B (float): Frequency.
        C (float): Phase shift.
        D (float): Vertical shift.

    Returns:
        np.ndarray | float: Sinusoidal output.
    """
    return A * np.sin(B * x + C) + D

def rotated_sinusoid(x, A, B, C, D, theta):
    """
    Rotates a sinusoidal function by a given angle.

    Args:
        x (np.ndarray | float): Input values.
        A, B, C, D (float): Sinusoidal parameters.
        theta (float): Rotation angle.

    Returns:
        np.ndarray | float: Rotated sinusoidal output.
    """
    y = A * np.sin(B * x + C) + D
    x_rot = x * np.cos(theta) - y * np.sin(theta)
    y_rot = x * np.sin(theta) + y * np.cos(theta)
    return y_rot


def fitSin(movementses, output_path, video_file_path):
    """
    Fits a rotated sinusoidal model to cumulative movements.

    Args:
        movementses (list[np.ndarray]): List of cumulative movements.
        output_path (str): Directory to save debug plots.
        video_file_path (str): Video file path for labeling outputs.

    Returns:
        np.ndarray: Fitted sinusoidal parameters.
    """
    paramses = []
    for i, movements in enumerate(movementses):
        x = np.arange(len(movements))
        max_m = np.max(movements)
        min_m = np.min(movements)
        freq = 2 * IMAGE_REPEATS * np.pi / len(movements)

        custom_rotated_sinusoid = lambda x, A, C, D, theta: rotated_sinusoid(x, A, freq, C, D, theta)

        # Initial guess for parameters: [Amplitude, Frequency, Phase, Vertical shift, Rotation]
        initial_guesses = [(max_m - min_m) / 2, 0, 0, 0]

        lower_bounds = [0, -np.pi, -100, -0.1]  # Set lower bounds
        upper_bounds = [max(max_m, -min_m), np.pi, +100, 0.1]  # Set upper bounds

        # Fit the model
        params, _ = curve_fit(custom_rotated_sinusoid, x, movements, p0=initial_guesses,
                              bounds=(lower_bounds, upper_bounds), method='trf', maxfev=5000)
        A, C, D, theta = params
        params = A, freq, C, D, theta
        if TESTING_MODE:
            # Extract fitted parameters
            # print(f"Amplitude: {A}, Frequency: {freq}, Phase: {C}, Vertical Shift: {D}, Theta: {theta}")

            # Generate fitted rotated sinusoid
            fitted_rotated_sinusoid = rotated_sinusoid(x, A, freq, C, D, theta)
            # Plot the results
            plt.figure(figsize=(10, 5))
            plt.plot(x, movements, label="Cumulative Movements")
            plt.plot(x, fitted_rotated_sinusoid, label="Fitted Rotated Sinusoid", linestyle="--")
            plt.legend()
            plt.xlabel("Stripe Index")
            plt.ylabel("Movement")
            plt.title("Fitting Rotated Sinusoidal Model to Cumulative Movements")
            plt.grid(True)
            plt.savefig(os.path.join(output_path, f"{os.path.splitext(video_file_path)[0]}_oio_{i}_fitsin" + ".png"))

        paramses.append(params)

    return np.array(paramses)


def remove_sinusoidal_transformation(images, paramses):
    """
    Removes sinusoidal distortions from image rows.

    Args:
        images (list[np.ndarray]): List of image rows.
        paramses (np.ndarray): Sinusoidal parameters for correction.

    Returns:
        list[np.ndarray]: Corrected image rows.
    """
    rows = []
    # A, B, _, _, _ = np.median(paramses, axis=0)
    for image, params in zip(images, paramses):
        A, B, C, _, _ = params

        # Create an empty output image
        output_image = np.zeros_like(image)

        # Track the maximum shift
        max_shift = 0

        # Loop over each column
        for j in range(image.shape[1]):
            # Calculate the vertical shift for this column based on the sinusoidal function
            shift = sinusoid(j, A, B/SINUSOID_SAMPLING, C, 0)
            max_shift = max(max_shift, abs(shift))  # Update maximum shift

            # Shift the whole column
            # Use np.roll to shift the column by the calculated value
            new_column = np.roll(image[:, j], int(shift))

            # Assign the shifted column back to the output image
            output_image[:, j] = new_column

        # Crop the image to remove the wrapped-around pixels
        if max_shift > 0:
            output_image = output_image[int(max_shift): -int(max_shift), :]

        rows.append(output_image)

    return rows


def remove_sin_transform(rows, moving_down, output_path, video_file_path):
    """
    Wrapper function to remove sinusoidal distortions.

    Args:
        rows (list[np.ndarray]): List of image rows.
        moving_down (bool): Indicates downward movement.
        output_path (str): Directory to save debug plots.
        video_file_path (str): Video file path for labeling outputs.

    Returns:
        list[np.ndarray]: Corrected image rows.
    """
    if rows:
        movementses = calculate_movements(rows, moving_down)
        movementses = detrend_movements(movementses, output_path, video_file_path)
        params = fitSin(movementses, output_path, video_file_path)
        print(f"\nParameters for sinusoidal transformation:\n{params}\n")
        rows = remove_sinusoidal_transformation(rows, params)

    return rows



def construct_rows(motions, intervals, video_file_path, output_path):
    """
    Constructs image rows by processing video frames.

    Args:
        motions (VideoMotion): VideoMotion instance for motion analysis.
        intervals (list): Detected motion intervals.
        video_file_path (str): Path to the video file.
        output_path (str): Directory to save output rows.

    Returns:
        None
    """
    print(f"Processing RowBuilder for: {video_file_path}\n")
    vidcap = cv2.VideoCapture(video_file_path)
    rows = []
    for i, interval in enumerate(intervals):
        mn, mx = interval
        start = mn + (mx - mn) // 2 - motions.get_frames_per360() // 2
        end = start + motions.get_frames_per360()
        file_path = os.path.join(output_path, os.path.splitext(os.path.basename(video_file_path))[0] + f"-oio-{i}.png")
        if not os.path.isfile(file_path):
            row = construct_row(vidcap, int(start), int(end), direction=motions.get_direction(),
                                rotation=motions.is_portrait(), shift_per_frame=motions.get_horizontal_speed(),
                                frames_per_360_deg=motions.get_frames_per360())
            rows.append(row)

    rows = remove_sin_transform(rows, motions.is_moving_down(), output_path, video_file_path)

    for i, row in enumerate(rows):
        file_path = os.path.join(output_path, os.path.splitext(os.path.basename(video_file_path))[0] + f"-oio-{i}.png")
        iio.imwrite(file_path, row.astype(np.uint8))


def construct_row(vidcap,
                  start: int,
                  end: int,
                  shift_per_frame,
                  frames_per_360_deg,
                  direction: str,
                  rotation: bool = False,
                  blended_pixels_per_frame=BLENDED_PIXELS_PER_FRAME,
                  blended_pixels_shift=BLENDED_PIXELS_SHIFT):
    """
    Constructs a single row image from video frames.

    Args:
        vidcap (cv2.VideoCapture): Video capture object.
        start (int): Starting frame index.
        end (int): Ending frame index.
        shift_per_frame (float): Horizontal shift per frame.
        frames_per_360_deg (int): Number of frames for a 360-degree rotation.
        direction (str): Motion direction ("CCW" or "CW").
        rotation (bool): Whether to rotate frames.
        blended_pixels_per_frame (int): Pixels blended per frame.
        blended_pixels_shift (int): Shift for blending.

    Returns:
        np.ndarray: Constructed row image.
    """
    if not vidcap or end - start <= 0:
        raise IOError("Unsupported input data.")

    frame_size = (int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # pixel intensity accumulator
    frame_shift_to_pixels_total = math.ceil((frames_per_360_deg + blended_pixels_per_frame) * shift_per_frame)
    # frame_shift_to_pixels_total = np.ceil((end-start) * shift_per_frame + blended_pixels_per_frame).astype(int)
    # print(frame_shift_to_pixels_total, shift_per_frame)
    row_image = np.zeros(
        (frame_size[0],
         frame_shift_to_pixels_total))
    # weight matrix for the accumulator
    weight_matrix = np.zeros(row_image.shape)

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start - blended_pixels_per_frame//2)

    if rotation:
        image_part = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    else:
        image_part = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    for frameNo in tqdm(range(0, int(end - start + blended_pixels_per_frame//2)), desc="Building row image"):
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
            crop_x_start = (math.floor(row_image.shape[1] - shift_per_frame * frameNo - blended_pixels_per_frame)
                            if direction == "CCW"
                            else math.floor(shift))
            crop_x_end = np.max([0, crop_x_start + blended_pixels_per_frame])
            # print(crop_x_start, crop_x_end, row_image.shape)
            # print(image[:, (image_part // 2 - blended_pixels_per_frame // 2) + blended_pixels_shift: (image_part // 2 + blended_pixels_per_frame // 2) + blended_pixels_shift, 0].shape)
            # print(aligned_image[:, (image_part // 2 - blended_pixels_per_frame // 2) + blended_pixels_shift: (image_part // 2 + blended_pixels_per_frame // 2) + blended_pixels_shift, 0].shape)
            row_image[:, crop_x_start:crop_x_end] += aligned_image[:, (image_part // 2 - blended_pixels_per_frame // 2) + blended_pixels_shift: (image_part // 2 + blended_pixels_per_frame // 2) + blended_pixels_shift, 0]
            weight_matrix[:, crop_x_start:crop_x_end] += 1
        else:
            print("wrong status")

    row_image = (row_image / weight_matrix)[:, math.ceil(blended_pixels_per_frame * shift_per_frame):-math.ceil(blended_pixels_per_frame * shift_per_frame)]

    return np.copy(row_image)
