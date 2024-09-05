import os
import subprocess


def _dump_path(video_file_path):
    return video_file_path.replace(".MP4", "-video-info.txt")


def extract_and_dump(video_file_path):
    # taken from palivo - list timing of the video
    _cmd = ["ffprobe",
            "-i", str(video_file_path),
            "-show_frames",
            "-show_entries", "frame=pkt_dts_time",
            "-select_streams", "v:0",
            "-print_format", "flat"]
    video_info_file_path = _dump_path(video_file_path)
    with open(video_info_file_path, 'w') as f:
        p1 = subprocess.run(_cmd, stdout=f)


def load_or_extract(video_file_path: str):
    """
    :param video_file_path: path to source video file
    :return: list of tuples[frameNo, timestamp]
    """
    video_timing = []
    if not os.path.isfile(_dump_path(video_file_path)):
        extract_and_dump(video_file_path)
    with open(_dump_path(video_file_path), "rt") as f:
        video_timing = f.readlines()

    timing = []
    for vt in video_timing:
        single_line = vt.rstrip("\n")
        timestamp = float(single_line.split('"')[1])
        frame_no = int(single_line.split(".")[2])
        timing.append((frame_no, timestamp))

    return timing
