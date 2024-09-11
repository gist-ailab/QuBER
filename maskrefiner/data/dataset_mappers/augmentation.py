import random
import numpy as np
from fvcore.transforms import transform, Transform
from detectron2.data.transforms import RandomCrop, StandardAugInput
from detectron2.data.transforms.augmentation import Augmentation
from detectron2.structures import BoxMode

from PIL import Image
import pyfastnoisesimd as fns


def perlin_noise(frequency, width, height):

    noise = fns.Noise()
    noise.NoiseType = 2 # perlin noise
    noise.frequency = frequency
    result = noise.genAsGrid(shape=[height, width], start=[0,0])
    return result

def PerlinDistortion(image):
    """
    """
    height, width = image.shape
    # sample distortion parameters from noise vector
    fx = np.random.uniform(0.0001, 0.1)
    fy = np.random.uniform(0.0001, 0.1)
    fz = np.random.uniform(0.01, 0.1)
    wxy = np.random.uniform(0, 10)
    wz = np.random.uniform(0, 0.005)
    cnd_x = wxy * perlin_noise(fx, width, height)
    cnd_y = wxy * perlin_noise(fy, width, height)
    cnd_z = wz * perlin_noise(fz, width, height)

    cnd_h = np.array(list(range(height)))
    cnd_h = np.expand_dims(cnd_h, -1)
    cnd_h = np.repeat(cnd_h, width, -1)
    cnd_w = np.array(list(range(width)))
    cnd_w = np.expand_dims(cnd_w, 0)
    cnd_w = np.repeat(cnd_w, height, 0)

    noise_cnd_h = np.int16(cnd_h + cnd_x)
    noise_cnd_h = np.clip(noise_cnd_h, 0, (height - 1))
    noise_cnd_w = np.int16(cnd_w + cnd_y)
    noise_cnd_w = np.clip(noise_cnd_w, 0, (width - 1))

    new_img = image[(noise_cnd_h, noise_cnd_w)]
    new_img = new_img = new_img + cnd_z
    return new_img.astype(np.float32)