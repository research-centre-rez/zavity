# OIO Pipeline

## Description
Makes One Image Overview (OIO) from video scan of threaded insert.

## Configuration

### Configuration file
Before running, navigate to configuration file located at `/src/config/config.py`. Define paths to input and output folder and other configurable parameters, mostly to adapt it to your environment.

### Main Method Arguments
- `--mode {single,multiple}`: Specify the mode of operation (single video or multiple videos). If `single` you have to use argument `--video_name`. If `multiple` all files from `INPUT_FOLDER` (which is specified in configuration file `config.py`) are processed as videos.
- `--video_name`: (Required if using `single` mode) Path to the video file.
- `--calc_rot_per_frame`: (Default is False) Set to True to calculate its own rotation per frame, not using the precalculated one. Not recommended since it greatly increases computational time.

## Running
### Docker Usage
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
docker run -v /path/to/input:/input -v /path/to/output:/output -v /path/to/config/config.py:/app/src/config/config.py oio-pipeline --mode single --video_name "video_name.mp4"
```

### Manual Usage
1. Download and install FFmpeg

2. Install requirements, for example with pip:
```bash
pip install -r requirements.txt
```
3. Run the main script, for example:
```bash
python src/scripts/main.py --mode single --video_name "video_name.mp4"
```
