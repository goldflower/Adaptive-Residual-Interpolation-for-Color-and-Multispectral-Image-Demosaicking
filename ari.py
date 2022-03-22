import numpy as np
import numpy.matlib
from green_interpolation import green_interpolation
from red_blue_interpolation_first import red_blue_interpolation_first
from red_blue_interpolation_second import red_blue_interpolation_second

def mosaic_bayer_bggr(rgb_1d):
  mask_red = np.zeros_like(rgb_1d)
  mask_green1 = np.zeros_like(rgb_1d)
  mask_green2 = np.zeros_like(rgb_1d)
  mask_blue = np.zeros_like(rgb_1d)
  mask_red[1::2, 1::2] = 1
  mask_green1[::2, 1::2] = 1
  mask_green2[1::2, ::2] = 1
  mask_green = mask_green1 + mask_green2
  mask_blue[::2, ::2] = 1

  red = np.zeros(rgb_1d.shape)
  green = np.zeros(rgb_1d.shape)
  blue = np.zeros(rgb_1d.shape)
  red = rgb_1d * mask_red
  green = rgb_1d * mask_green
  blue = rgb_1d * mask_blue

  mosaic = np.concatenate([red[..., None], green[...,None], blue[..., None]], axis=-1)
  mask = np.concatenate([mask_red[..., None], mask_green[..., None], mask_blue[..., None]], axis=-1)
  return mosaic, mask


def demosaic_ARI(rgb, skip_mosaic=False, mask=None):
  # only support "BGGR"

  eps = 1e-10
  if not skip_mosaic:
    mosaic_input, mask = mosaic_bayer_bggr(rgb)
    # mosaic_input = mosaic_input[1024:2048, 2048:3072, :]
    # mask = mask[1024:2048, 2048:3072, :]
    # io.imsave("test_mosaic.png", mosaic_input)
  else:
    mosaic_input = rgb
  if mask is None:
    raise ValueError("mask should not be None.")
  green = green_interpolation(mosaic_input, mask, eps)
  red, blue = red_blue_interpolation_first(green, mosaic_input, mask, eps)
  red, blue = red_blue_interpolation_second(green, red, blue, mask, eps)
  # red and blue interpolation (ICIP2015 version)
  # h = 5; v = 5; % guided filter window size
  # red  = red_interpolation(green, mosaic, mask, h, v, eps)
  # blue = blue_interpolation(green, mosaic, mask, h, v, eps)

  result = np.concatenate([red[..., None], green[..., None], blue[..., None]], axis=-1)
  return result
