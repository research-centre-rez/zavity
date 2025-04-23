# Dynamic paths based on the project directory
OUTPUT_FOLDER = "../output"  # Output folder at the same level as zavity
INPUT_FOLDER = "../input"  # Input folder at the same level as zavity

TESTING_MODE = True
VERBOSE = True
N_CPUS = 80

# VideoPreprocessor config
REMOVE_ROTATION = True
PREPROCESSOR_SAMPLING = 28  # min(2*14x+21529/x) min(2*(num_of_border_breakpoints+1)*sampling+total_frames/sampling)
# window coordinates
Y1 = 550  #Y1-PADDING cant be lower then 0
Y2 = 1850  #Y2+PADDING cant be bigger then 2160
X1 = 1350  #X1-PADDING cant be lower then 0
X2 = 2650  #X2+PADDING cant be bigger then 3840
PADDING = 300
PREPROCESSOR_DOWNSCALE = 4
SEGMENT_TYPE_TH = 0.04
ROT_PER_FRAME = 0.12733934740581357  # 0.12735436683938242  # 0.12734272843513264
RECTIFY = True
CALIBRATION_CONFIG_FILE_PATH = 'src/config'
CODEC = 'X264'  # 'mp4v' 'MJPG'
EXT = '.mp4'  # 'mp4v' '.avi'
INTERVAL_FILTER_TH = 0.25
LOAD_VIDEO_TO_RAM = True  # at least 256 GB RAM needed, but do not change, not implemented for False
PITCH_ANGLE = 3.43  # degrees

# VideoMotion config
MOTION_SAMPLING = 1
MOTION_DOWNSCALE = 4
ROW_ROTATION_OVERLAP_RATIO = 1.055  # Precalculated average ratio of mirror rotation per row

# RowBuilder config
BLENDED_PIXELS_PER_FRAME = 11  # number of pixels taken from one image (it is volatile and it must be greater than SHIFT_PER_FRAME)
BLENDED_PIXELS_SHIFT = 0  # number of pixels to shift area taken from one image
SINUSOID_SAMPLING = 10
IMAGE_REPEATS = 10

# RowStitcher config
SEARCH_SPACE_SIZE = (36, 36)  # of shift correction between pairs of row images (roll, shift)
XTOL = 1e-3
FTOL = 1e-5
