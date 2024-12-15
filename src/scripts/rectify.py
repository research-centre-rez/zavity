import argparse
import cv2
from tqdm.auto import tqdm
import steps.checkerboard_corners as checkerboard_corners
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import json
import sys

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _save_calibration_parameters(calibration_parameters, output_path):
    """
    Saves camera calibration parameters to a specified output directory.

    :param calibration_parameters : tuple
        A tuple containing the calibration parameters: ret, mtx, newcameramtx, dist, rvecs, tvecs (opencv format).
    :param output_path : str
        The directory path where the calibration file will be saved. Must exist prior to calling this function.

    :returns : dict
        A dictionary containing the calibration parameters that were saved, structured with keys:
        'ret', 'mtx', 'newcameramtx', 'dist', 'rvecs', and 'tvecs'.

    Raises : AssertionError If the specified output path does not exist.
    Notes: The calibration parameters are serialized and written to a file in JSON format using a custom encoder for
        handling numpy data types. Logging is used to record the successful saving of calibration data.
    """
    assert os.path.isdir(output_path), f"Output path {output_path} not found. Please create it first."
    ret, mtx, newcameramtx, dist, rvecs, tvecs = calibration_parameters
    cam_calib = {
        "ret": ret,
        "mtx": mtx,
        "newcameramtx": newcameramtx,
        "dist": dist,
        "rvecs": rvecs,
        "tvecs": tvecs
    }
    json.dump(cam_calib, open(_checkerboard_calibration_filepath(output_path), "wt"), cls=NumpyEncoder)
    logger.info(f"Calibration saved in {_checkerboard_calibration_filepath(output_path)}.")
    return cam_calib


def _load_calibration_parameters(calibdir_path):
    """
    Loads camera calibration parameters from a specified directory.

    The function reads a JSON file containing the camera calibration data
    from the given directory and extracts the calibration parameters. It
    returns these parameters as a tuple consisting of:
    - the reprojection error
    - the camera matrix
    - the new camera matrix
    - the distortion coefficients
    - the rotation vectors
    - the translation vectors.

    Parameters:
    - calibdir_path: Path to the directory containing the calibration file.

    Returns:
    - A tuple containing the reprojection error, camera matrix, new camera matrix,
      distortion coefficients, rotation vectors, and translation vectors.
    """
    cam_calib = json.load(open(_checkerboard_calibration_filepath(calibdir_path), "rt"))
    ret = cam_calib["ret"]
    mtx = np.array(cam_calib["mtx"])
    newcameramtx = np.array(cam_calib["newcameramtx"])
    dist = np.array(cam_calib["dist"])
    rvecs = np.array(cam_calib["rvecs"])
    tvecs = np.array(cam_calib["tvecs"])

    return ret, mtx, newcameramtx, dist, rvecs, tvecs


def _rectified_video_path(input_video_path, output_path):
    """
    Generates the path for the rectified video file based on the input video file path.

    Parameters:
    output_path (str): The directory where the rectified video file should be stored.
    input_video_path (str): The original video file path that needs rectification.

    Returns:
    str: The file path for the rectified video, replacing the ".MP4" extension with "-rectified.MP4".
    """
    # TODO: adapt for non MP4 files
    basename = os.path.basename(input_video_path)
    suffix = basename.split(".")[-1]
    return os.path.join(output_path, basename.replace(f".{suffix}", f"-rectified.{suffix}"))


def _checkerboard_calibration_filepath(prefix):
    """
    Constructs and returns the file path for the checkerboard calibration file.

    The function takes in a directory path prefix and appends the filename
    'checkerboard-calibration.json' to this prefix. This constructs an
    absolute or relative file path, depending on the nature of the provided prefix.

    Parameters:
    prefix (str): The directory path prefix where the file is located.

    Returns:
    str: The complete file path including the filename 'checkerboard-calibration.json'.
    """
    return os.path.join(prefix, "checkerboard-calibration.json")


def print_corner_count_histogram(point_pairs_list):
    """
    Generates and displays (or saves) a histogram depicting the number of reference points found in a series of frames.

    :param point_pairs_list : list
        A list of tuples where each tuple contains a frame number, a frame, a list of object points, and a list of
        image points. The length of the object points list for each frame is used to create the histogram.

    Behavior:
    - The function creates a histogram showing the distribution of the number of reference points found across different frames.
    - The x-axis of the histogram represents the number of reference points found in a frame.
    - The y-axis represents the number of frames that contain a particular number of reference points.
    - If an output path is specified by the global variable OUTPUT_PATH, the histogram is saved to a file named "corner_count_histogram.png" in that directory.
    - If OUTPUT_PATH is not specified, the histogram is displayed directly to the user.
    """
    plt.figure(figsize=(10, 5))
    plt.hist([len(objpoints) for frame_no, frame, objpoints, imgpoints in point_pairs_list])
    plt.title("Reference points found")
    plt.xlabel("# reference points")
    plt.ylabel("# frames")
    if OUTPUT_PATH is not None:
        plt.savefig(os.path.join(OUTPUT_PATH, "corner_count_histogram.png"))
    else:
        plt.show()
    plt.close()


def find_point_pairs(cap, frame_no_range=None):
    """
    Find checkerboard corners in the video.
    :param cap: OpenCV video capture object
    :param frame_no_range: tuple of (start, end) frame numbers to process. If None, process all frames (time consuming)

    :return point_pairs_list: list of tuples (frame_no, frame, objpoints, imgpoints) where objpoints and imgpoints are
        the corners found in the image.
    """

    if frame_no_range is None:  # as default process all frames in the video
        frame_no_range = (0, cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no_range[0])
    # TODO: this is black magic and should be moved into configuration
    np.random.seed(1234)
    frame_nos = np.random.randint(frame_no_range[0], frame_no_range[1], 100)
    # frame_nos = np.arange(frame_no_range[0], frame_no_range[1] / 5, 20)

    point_pairs_list = []
    for frame_no in tqdm(frame_nos, desc="Looking for corners"):
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            success, frame = cap.read()
            if success:
                objpoints, imgpoints = checkerboard_corners.process_image(frame, 6)
                if objpoints.size >= 10 * 3:
                    point_pairs_list.append((frame_no, frame, objpoints, imgpoints))
                    logger.debug(f"Frame {frame_no}: {len(objpoints)}")
            else:
                logger.error(f"Unexpected end of video. Expected about {frame_nos - frame_no} more frames.")
                break
        except Exception as e:
            logger.warning(f"General exception at {frame_no}: {e}")
    return point_pairs_list


def subpix_corners(output_path, point_pairs_list, disable_plot=False):
    """
    Refines the corner locations for a list of images using the sub-pixel corner detection method.

    This function takes a list of image data where corners of a checkerboard pattern have been identified.
    For each set of corner points, it refines the corner locations to sub-pixel accuracy using OpenCV's
    cornerSubPix function. If plotting is not disabled, it saves the annotated images with original and
    refined corner locations into a specified output path for manual verification.

    Parameters:
        output_path (str): The directory path where images with annotated corner points will be saved.
        point_pairs_list (list): A list of tuples containing frame number, image frame, object points,
                                 and image points for each image analyzed.
        disable_plot (bool): A flag indicating whether to disable plotting and saving of images.
                             Default is False.

    Returns:
        list: A list containing tuples of frame number, image frame, object points, and the recomputed
              image points with sub-pixel accuracy.
    """
    subpix_corners = []
    for frameno, frame, objpoints, imgpoints in point_pairs_list:
        recomputed_corners = cv2.cornerSubPix(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                      imgpoints.astype(np.float32),
                                      (15, 15),
                                      (-1, -1),
                                      (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001))
        subpix_corners.append((frameno, frame, objpoints, recomputed_corners))

    if disable_plot:
        return subpix_corners

    # Dump images into folder for manual check
    TMP = os.path.join(output_path, "checkerboard-corners")
    os.makedirs(TMP, exist_ok=True)
    for (frameno, frame, objpoints, imgpoints), (_, _, _, recomputed) in zip(point_pairs_list, subpix_corners):
        plt.figure(figsize=(20,11))
        plt.imshow(frame)
        plt.scatter(imgpoints[:,0], imgpoints[:, 1], color="red", marker="o", facecolors='none')
        plt.scatter(recomputed[:,0], recomputed[:, 1], color="blue", marker="o", facecolors='none')
        plt.tight_layout()
        plt.savefig(os.path.join(TMP, f"{frameno.astype(int):03d}.png"))
        plt.close()

    return subpix_corners


def calibration_parameters(subpix_corners, frame_shape=(2160, 3840)):
    """
    Calculates the camera calibration parameters using detected corners and frame shape.

    Parameters:
    subpix_corners : list
        A list of tuples containing detected corner points for each calibration pattern.
        Each tuple consists of subpixel corner coordinates, pattern size, object points, and image points.
    frame_shape : tuple, optional
        The shape of the frame (height, width), by default GoPro max resolution (2160, 3840).

    Returns:
    tuple
        A tuple containing the following calibration parameters:
        - ret: The overall RMS re-projection error.
        - mtx: Camera matrix.
        - newcameramtx: New camera matrix based on a free scaling parameter.
        - dist: Distortion coefficients.
        - rvecs: Rotation vectors estimated for each pattern view.
        - tvecs: Translation vectors estimated for each pattern view.
    """
    h, w = frame_shape

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        [objpoints.astype(np.float32) for _, _, objpoints, _ in subpix_corners],
        [imgpoints.astype(np.float32) for _, _, _, imgpoints in subpix_corners],
        (w, h),
        None,
        None
    )
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    return ret, mtx, newcameramtx, dist, rvecs, tvecs


def convert_video(video_file_path, calib_params, output_file_path):
    """
    Processes a video file to remove lens distortion and saves it to a new file using specified calibration parameters.

    Parameters:
        video_file_path (str): The path to the input video file that needs to be processed.
        calib_params (tuple): A tuple containing calibration parameters, typically including
                              the camera matrix and distortion coefficients.
        output_file_path (str): The path where the processed video will be saved.

    The function utilizes OpenCV to read an input video, apply corrections based on the calibration
    parameters to remove lens distortion, and then writes the corrected frames to a new video file.
    It reads the frame rate and frame count of the source video to ensure the output video maintains
    the same timing characteristics. The output video is encoded using the H264 codec.
    The function logs an error if it fails to read the input video.
    """
    _, mtx, newcameramtx, distortion, _, _ = calib_params

    vidcap = cv2.VideoCapture(video_file_path)
    frame_count = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = vidcap.get(cv2.CAP_PROP_FPS)

    success, frame = vidcap.read()
    if not success:
        logger.error(f"Could not read from the videofile {video_file_path}")
        return

    w, h = frame.shape[:2][::-1]
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, distortion, None, newcameramtx, (w, h), 5)
    logger.info(f"Writing video to {output_file_path}")
    out = cv2.VideoWriter(output_file_path,
                          apiPreference=cv2.CAP_FFMPEG,
                          fourcc=cv2.VideoWriter_fourcc(*"H264"),
                          fps=fps,
                          # It is crucial to properly setup header of the writer otherwise the file will be empty
                          frameSize=(w, h),  # doublecheck that the frames has same size as is written here
                          params=[
                              cv2.VIDEOWRITER_PROP_DEPTH,
                              cv2.CV_8U,  # Format must match out.write data
                              cv2.VIDEOWRITER_PROP_IS_COLOR,  # If you write RGB images this must be 1 (grayscale = 0)
                              1,  # True
                          ])

    for _ in tqdm(np.arange(frame_count), desc="Frame rectification"):
        success, frame = vidcap.read()
        if not success:
            logger.error(f"Could not read from the videofile {video_file_path}")
            return
        undistorted = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
        out.write(undistorted.astype(np.uint8))  # doublecheck that the frames has same size as is written in the header
    out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    Tool for video rectification
    """)
    parser.add_argument(
        "-ic",
        "--checkerboard-input-video-path",
        type=str,
        required=False,
        help="Path to the input video with the checkerboard"
    )

    parser.add_argument(
        "-cc",
        "--calibration-config-dir",
        type=str,
        required=False,
        help="Path to the calibration config file"
    )

    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        required=True,
        help="Path to the output folder"
    )

    parser.add_argument(
        "-i",
        "--input-video-path",
        type=str,
        required=False,
        help="Path to the video which should be converted"
    )

    parser.add_argument(
        "--logging-output",
        type=str,
        required=False,
        help="Path to the output dir for logging"
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    if args.logging_output:
        OUTPUT_PATH = args.logging_output
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(OUTPUT_PATH, "rectify.log")),
            ])
    else:
        OUTPUT_PATH = None
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)]
        )

    os.makedirs(args.output_path, exist_ok=True)
    if args.calibration_config_dir is None:
        if args.checkerboard_input_video_path is not None:
            checkerboard_cap = cv2.VideoCapture(args.checkerboard_input_video_path)
            point_pairs_list = find_point_pairs(checkerboard_cap)
            subpix_corners_list = subpix_corners(args.output_path, point_pairs_list, disable_plot=not args.verbose)
            if args.verbose:
                print_corner_count_histogram(point_pairs_list)

            calib_params = calibration_parameters(subpix_corners_list)
            _save_calibration_parameters(calib_params, args.output_path)
        else:
            logger.error("Either checkerboard input video path or calibration config must be provided.")
    else:
        calib_params = _load_calibration_parameters(args.calibration_config_dir)
        logger.info(
            f"Loaded calibration parameters from {args.calibration_config_dir}."
        )

    if args.input_video_path is not None:
        convert_video(
            video_file_path=args.input_video_path,
            calib_params=calib_params,
            output_file_path=_rectified_video_path(args.input_video_path, args.output_path)
        )


