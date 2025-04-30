import argparse
import logging
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "..")
sys.path.append(src_dir)

from timeit import default_timer as timer
from contextlib import contextmanager
from config.config import OUTPUT_FOLDER, INPUT_FOLDER
from steps.image_row_builder import ImageRowBuilder
from steps.image_row_stitcher import ImageRowStitcher
from steps.video_camera_motion import VideoMotion


@contextmanager
def timing(name):
    start = timer()
    yield
    end = timer()
    duration = end - start
    message = f"{name}: {duration:.4f} seconds"
    logging.debug(message)


def configure_logging(filename):
    log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # File handler
    file_handler = logging.FileHandler(os.path.join(OUTPUT_FOLDER, f"{filename}.log"))
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.INFO)

    # Add handlers manually instead of basicConfig
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger().addHandler(file_handler)
    logging.getLogger().addHandler(console_handler)


def process_single_video(video_name, calc_rot_per_frame):
    video_path = os.path.join(INPUT_FOLDER, video_name)
    configure_logging(video_name)
    logging.info(f"Processing single video: {video_path}. Output will be saved to {OUTPUT_FOLDER}.")
    process_video(video_path, calc_rot_per_frame)


def process_multiple_videos(folder_path, calc_rot_per_frame):
    configure_logging(os.path.basename(folder_path))
    logging.info(f"Processing multiple videos in folder: {folder_path}. Output will be saved to {OUTPUT_FOLDER}.")
    # List all video files in the specified folder
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(".mp4"):
            continue
        configure_logging(filename)
        logging.info(f"Processing video: {filename}.")
        video_path = os.path.join(folder_path, filename)
        if os.path.isfile(video_path):
            try:
                process_video(video_path, calc_rot_per_frame)
            except Exception as e:
                logging.error(f"Error processing video {filename}: {e}")


def process_video(video_path, calc_rot_per_frame):
    with timing("Total OIO Pipeline"):
        # Pipeline stages
        with timing("Preprocessor"):
            from steps.video_preprocessor import VideoPreprocessor
            preprocessor = VideoPreprocessor(video_path, calc_rot_per_frame)
            preprocessor.process()
            video_file_path = preprocessor.get_output_video_file_path()
            frames = preprocessor.getProcessedFrames()

        with timing("VideoMotion"):
            motions = VideoMotion(frames, video_file_path, preprocessor.get_intervals())
            motions.process()

        with timing("RowBuilder"):
            constructor = ImageRowBuilder(frames, motions, preprocessor.get_intervals(), video_file_path)
            rows = constructor.construct_rows()

        with timing("RowStitcher"):
            stitcher = ImageRowStitcher(rows, motions, video_path)
            stitcher.process()

    logging.info("OIO done")


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process single or multiple videos into one image overview.")
    parser.add_argument("--mode", type=str, choices=["single", "multiple"], required=True,
                        help="Mode of processing: 'single' for one video, 'multiple' for a folder of videos")
    parser.add_argument("--video_name", type=str, help="Name of the video located in INPUT_FOLDER to process")
    parser.add_argument("--calc_rot_per_frame", type=bool, default=False,
                        help="Set to True to calculate its own rotation per frame, not using the precalculated one."
                             "It takes around 2 hours. Also it compares the precalculated one with calculated one.")

    # Parse the arguments
    args = parser.parse_args()

    # Process based on mode
    if args.mode == "single":
        if not args.video_name:
            raise ValueError("Please provide --path_to_video for single video processing mode.")
        process_single_video(args.video_name, args.calc_rot_per_frame)

    elif args.mode == "multiple":
        process_multiple_videos(INPUT_FOLDER, args.calc_rot_per_frame)
