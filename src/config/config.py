import os

# Get the parent directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory of the config folder
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", ".."))  # Navigate up to the zavity folder

# Dynamic paths based on the project directory
OUTPUT_FOLDER = os.path.join(PROJECT_DIR, 'output')  # Output folder at the same level as zavity
INPUT_FOLDER = os.path.join(PROJECT_DIR, 'input')  # Input folder at the same level as zavity

VERBOSE = True

# VideoPreprocessor config
REMOVE_ROTATION = True
EVERY_NTH_FRAME = 28  # min(2*14x+21529/x) min(2*(num_of_border_breakpoints+1)*x+total_frames/x)
# window coordinates
Y1 = 550 #700    # Y1-PADDING cant be lower then 0
Y2 = 1850 #1900   # Y2+PADDING cant be bigger then 2160
X1 = 1350 #1500   # X1-PADDING cant be lower then 0
X2 = 2650 #2700   # X2+PADDING cant be bigger then 3840
PADDING = 300
DOWNSCALE = 4
SEGMENT_TYPE_THRESHOLD=0.04
ROT_PER_FRAME = 0.12735436683938242 # 0.12734272843513264
RECTIFY = True
CODEC = 'mp4v' # 'MJPG'
EXT = '.mp4' # '.avi'
BORDER_BPS = None # For testing purposes [[0, 46], [3034, 3164]]
REFINED_BP = None # For testing purposes 1208
INTERVAL_FILTER_TH = 0.25
CALIBRATION_CONFIG_FILE_PATH = 'src/config'

# VideoMotion config
FPS_REDUCTION = 1
RESOLUTION_DECS = 4
ROW_OVERLAP = 1.0595

# RowBuilder config
BLENDED_PIXELS_PER_FRAME = 10  # number of pixels taken from one image (volatile, must be greater than SHIFT_PER_FRAME)
BLENDED_PIXELS_SHIFT = 0  # number of pixels to shift area taken from one image
SINUSOID_SAMPLING = 10
IMAGE_REPEATS = 10

# RowStitcher config
SEARCH_SPACE_SIZE = (22, 16) # of shift correction between pairs of row images (roll, shift)
TESTING_MODE = True # replaces border pixels of rows with 0s and does not blend
XTOL = 1e-1
FTOL = 1e-3
