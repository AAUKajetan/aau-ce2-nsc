#-----------------------COFNIG---------------------------------------------
RESOLUTION_RAMP_UP = [256, 512, 1024, 2048, 4096, 8192]  # Different resolutions to test

CHUNK_SIZE = 128  # Size of chunks for multiprocessing

YMIN, YMAX = -1.5, 1.5
XMIN, XMAX = -2.0, 1.0
MAX_ITER = 100
#for basic tests
WIDTH, HEIGHT = 1024, 1024

RESULT_DIR = "results/"