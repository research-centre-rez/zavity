import os

# Get the parent directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory of the config folder
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", ".."))  # Navigate up to the zavity folder

# Dynamic paths based on the project directory
OUTPUT_FOLDER = os.path.join(PROJECT_DIR, 'output')  # Output folder at the same level as zavity
INPUT_FOLDER = os.path.join(PROJECT_DIR, 'input')  # Input folder at the same level as zavity

# CONSTANTS

# VideoPreprocessor config
ROT_PER_FRAME = 0.12734514911853811 # 0.12711575238356873
# window coordinates
Y1 = 550 #700    # Y1-PADDING cant be lower then 0
Y2 = 1850 #1900   # Y2+PADDING cant be bigger then 2160
X1 = 1350 #1500   # X1-PADDING cant be lower then 0
X2 = 2650 #2700   # X2+PADDING cant be bigger then 3840
PADDING = 300
SAMPLES = 1000
THRESHOLD_DISTANCE_FOR_BREAKPOINT_MERGE = 1
EVERY_NTH_FRAME = 40  # min(2*6x+20000/x) min(2*num_of_border_breakpoints*x+total_frames/x)
SIGMA = 3 # to denoise histogram of angles with gaussian
CODEC = 'mp4v' # 'MJPG'
EXT = '.mp4' # '.avi'
BORDER_BPS = None # For testing purposes [[0, 46], [3034, 3164]]
REFINED_BP = None # For testing purposes 1208
CALIBRATE = False
SINUSOID_SAMPLING = 10

# VideoMotion config
FPS_REDUCTION = 1
RESOLUTION_DECS = 6
ROW_OVERLAP = 1.057

# RowBuilder config
BLENDED_PIXELS_PER_FRAME = 10  # number of pixels taken from one image (volatile, must be greater than SHIFT_PER_FRAME)
BLENDED_PIXELS_SHIFT = 0  # number of pixels to shift area taken from one image

# RowStitcher config
SEARCH_SPACE_SIZE = (30, 50) # of shift correction between pairs of row images
