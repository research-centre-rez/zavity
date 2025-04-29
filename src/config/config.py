# Dynamic paths based on the project directory
OUTPUT_FOLDER = "../output"
INPUT_FOLDER = "../input"

# CONFIGURABLE
VERBOSE = True  # plotting and image saving for debugging purposes
TESTING_MODE = False  # testing mode does not blend rows during stitching and draws rectangles on margin, so they are
# visible, also print intermediate row images (pre_sin, rolled)
N_CPUS = 80  # specifies number of cpus used during parallelization
RECTIFY = True  # Rectification of len's distortion can be set as False to decrease computation time but slightly
# decrease quality of resulting OIO.
LOAD_VIDEO_TO_RAM = True  # If True, it greatly reduces computing time, but requires FFmpeg to be installed in your
# environment and 35 GB of RAM per minute of the video.
CODEC = 'mp4v'  # If LOAD_VIDEO_TO_RAM = True, this codec is used to encode video after preprocessing.
# It's configurable based on the environment. Other options: 'avc1', 'MJPG', etc.
EXT = '.mp4'  # If LOAD_VIDEO_TO_RAM = True, this extension is used to encode video after preprocessing.
# It's configurable based on the codec. Other options: '.avi', etc.

# CONFIGURABLE PHYSICS OF THREADED INSERT
PITCH_ANGLE = 3.43  # pitch angle of threaded insert in degrees

# NOT RECOMMENDED TO CHANGE - TUNED FOR CURRENT HARDWARE SETUP
# VideoPreprocessor config
REMOVE_ROTATION = True  # controls rotation removal during preprocessing
# it's there for historical reasons
PREPROCESSOR_SAMPLING = 28  # optimized at min(2*(num_of_border_breakpoints+1)*sampling+total_frames/sampling) e.g.
# min(2*14x+21529/x)
# cropping window coordinates
Y1 = 550  # Y1-PADDING cant be lower than 0
Y2 = 1850  # Y2+PADDING cant be bigger than height of input video (e.g. 2160)
X1 = 1350  # X1-PADDING cant be lower than 0
X2 = 2650  # X2+PADDING cant be bigger than width of input video (e.g. 3840)
PADDING = 300  # extra padding used during rotation compensation
PREPROCESSOR_DOWNSCALE = 4  # downscaling used during preprocessing
SEGMENT_TYPE_TH = 0.04  # threshold used to find out a segment type
ROT_PER_FRAME = 0.12735436683938242  # can be updated with recalculated rotation with --calc_rot_per_frame True
RECTIFICATION_PARAMS_FOLDER = 'src/config'  # path of folder which includes 'checkerboard-calibration.json' file
# including parameters needed for rectification, used only with RECTIFY = True
INTERVAL_FILTER_TH = 0.2  # threshold of a difference ratio in frame length to filter small intervals
# VideoMotion config
MOTION_SAMPLING = 1  # sampling used during motion detection
MOTION_DOWNSCALE = 4  # downscaling used during motion detection
ROW_ROTATION_OVERLAP_RATIO = 1.055  # Precalculated ratio of device's rotational movement, meaning it does full
# rotation and extra 5.5 % of circle
# RowBuilder config
BLENDED_PIXELS_PER_FRAME = 11  # number of columns taken from one frame to form row image
# (it is volatile, and it must be greater than rotational speed in pixels)
BLENDED_PIXELS_SHIFT = 0  # offset to shift area taken from one frame to form row image in pixels
SINUSOID_SAMPLING = 10  # sampling used during removal of sinusoidal error in row image construction
STRIPE_WIDTH = 100  # width of stripe used during removal of sinusoidal error in row image construction
IMAGE_REPEATS = 10  # number of repeats for image stacking to better fit sinusoidal error in row image construction
# RowStitcher config
SEARCH_SPACE_SIZE = (24, 36)  # Search space size used during optimizing shift correction between pairs of row images.
# Each number in the tuple means that given variable is bounded to be optimized in bounds which are that number far
# from initial values.
# The order is (shift, roll).
X_TOL = 1e-1  # stopping criteria for optimization, optimized variable has to change by this amount between iterations
F_TOL = 1e-2  # stopping criteria for optimization, function value has to change by this amount between iterations
