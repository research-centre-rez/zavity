# physical constants
SHIFT_PER_FRAME = 13.95  # number of pixels for which are shifted two consecutive frames (depends on camera speed)
BLENDED_PIXELS_PER_FRAME = 20  # number of pixels taken from one image (volatile, must be greater than SHIFT_PER_FRAME)
FRAME_SIZE = (2160, 3840)  # size of the frame (GoPro setup)
FRAMES_PER_360_DEG = 1946  # number of frames per 360 deg (depends on camera speed)
LIGHT_ON_OFF_FRAME_DISTANCE = 1000

# paths
SRC = '/Users/gimli/cvr/data/zavity/Cely_sken_3_110proc'
