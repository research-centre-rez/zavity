import os

# Get the parent directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory of the config folder
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", ".."))  # Navigate up to the zavity folder

# Dynamic paths based on the project directory
OUTPUT_FOLDER = os.path.join(PROJECT_DIR, 'output')  # Output folder at the same level as zavity
INPUT_FOLDER = os.path.join(PROJECT_DIR, 'input')  # Input folder at the same level as zavity

# CONSTANTS

# VideoPreprocessor config
ROT_PER_FRAME = 0.12734514911853811
# window coordinates
Y1 = 550    # Y1-PADDING cant be lower then 0
Y2 = 1850   # Y2+PADDING cant be bigger then 2160
X1 = 1350   # X1-PADDING cant be lower then 0
X2 = 2650   # X2+PADDING cant be bigger then 3840
PADDING = 300
SAMPLES = 1000
THRESHOLD_DISTANCE_FOR_BREAKPOINT_MERGE = 1
EVERY_NTH_FRAME = 40  # min(2*6x+20000/x) min(2*num_of_border_breakpoints*x+total_frames/x)
SIGMA = 3 # to denoise histogram of angles with gaussian

# VideoMotion config
FPS_REDUCTION = 1
RESOLUTION_DECS = 6

# RowBuilder config
BLENDED_PIXELS_PER_FRAME = 10  # number of pixels taken from one image (volatile, must be greater than SHIFT_PER_FRAME)
BLENDED_PIXELS_SHIFT = 0  # number of pixels to shift area taken from one image

# RowStitcher config
SEARCH_SPACE_SIZE = (30, 15) # of shift correction between pairs of row images
