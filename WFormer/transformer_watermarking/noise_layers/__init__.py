import random


def get_random_float(float_range: [float]):
	return random.random() * (float_range[1] - float_range[0]) + float_range[0]


def get_random_int(int_range: [int]):
	return random.randint(int_range[0], int_range[1])


from .identity import Identity
from .crop import Crop, Cropout, Dropout
from .gaussian_noise import GN
from .middle_filter import MF
from .gaussian_filter import GF
from .salt_pepper_noise import SP
from .jpeg import Jpeg, JpegSS, JpegMask, JpegTest
from .combined import Combined
from .grid_crop import Grid_Crop as GC
from .Adjust_Brightness import Adjust_Brightness as AB
from .quantization import Quantization as QT
from .Adjust_hue import Adjust_hue as AH
from .Adjust_contrast import Adjust_Contrast as AC
from .Adjust_Saturation import Adjust_Saturation as AS
from .jpeg_compression2000 import JpegCompression2000 as Jpeg2000
from .webp import WebP
from .affine import Affine

