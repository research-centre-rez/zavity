#!/bin/bash
EXEDIR=/storage/plzen1/home/blahafra/thesis/zavity/

cd $EXEDIR
module load ffmpeg/4.4.1
module load python/3.11.11
source zavity-venv/bin/activate

# Run the script
python src/scripts/main.py --mode single --video_name "srp_GX010027.MP4"
