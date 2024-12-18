# OIO Pipeline

## Description
Makes overall image output from video scan.

## Docker Usage
To build the application image:
```bash
docker build -t oio-pipeline .
```
To run the application using Docker, you need to specify the config file path and input and output directories. Use the `-v` flag to bind your local paths.
Examples:
```bash
docker run -v /path/to/input:/input -v /path/to/output:/output -v /path/to/config/config.py:/app/src/config/config.py oio-pipeline --mode multiple
```
```bash
docker run -v /path/to/input:/input -v /path/to/output:/output -v /path/to/config/config.py:/app/src/config/config.py oio-pipeline --mode single --path_to_video '/input/GX010009_cely zavit clona nahoru.MP4'
```

## Main Method Arguments
- `--mode {single,multiple}`: Specify the mode of operation (single video or multiple videos).
- `--path_to_video PATH_TO_VIDEO`: (Required if using `single` mode) Path to the video file.
- `--path_to_folder PATH_TO_FOLDER`: (Required if using `multiple` mode) Path to the folder containing videos. Application then takes every file in the folder and run it as a video. So you need to have ONLY videos in that folder.
- `--calc_rot_per_frame`: (Default is False) Set to True to calculate its own rotation per frame, not using the precalculated one. It takes around 2 hours. Also, it compares the precalculated one with calculated one.
