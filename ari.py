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


if __name__ == "__main__":
  from isp_utils import *
  from skimage import io
  import os
  # order = "RGGB"
  order = "BGGR"
  ccm = np.array([1.740101, -0.690924, -0.049177, -0.149781, 1.300080, -0.150298, 0.079116, -0.809069, 1.729953]).reshape((3, 3))
  red_gain, blue_gain, rgb_gain = 1.937500, 1.555664, 1.010000
  file_name = "../test_raws_and_meta/20220109010707_rawhdrdump_4096x3072_05.raw"
  file_name = "../test_z3_raws/20220130005418_rawhdrdump_4096x3072_00.raw"

  raw_data_np = np.fromfile(file_name, dtype=np.uint16)
  raw_data_np = raw_data_np.reshape((3072, 4096))
  
  # if order == "RGGB":
  #   test_rgb = (cv2.cvtColor(raw_data_np, cv2.COLOR_BayerBG2RGB) - 64) / 959.  # RGGB
  # if order == "BGGR":
  #   test_rgb = (cv2.cvtColor(raw_data_np, cv2.COLOR_BayerRG2RGB) - 64) / 959.  # BGGR
  normed_raw = ((raw_data_np - 64) / 959.).clip(0,1)
  del raw_data_np
  for i in range(0, 3072, 1024):
    for j in range(0, 4096, 1024):
      patch_num = (i//1024+1)*(j//1024) + 1
      print("patch {} / {}".format(patch_num, 3072*4096//1024**2))
      patch_rgb = demosaic_ARI(normed_raw[i:i+1024, j:j+1024], False)
      patch_rgb = isp(patch_rgb, red_gain, blue_gain, rgb_gain, ccm)
      patch_rgb = (patch_rgb * 255).numpy().astype(np.uint8)[0]
      io.imsave("{}_{}.png".format(os.path.basename(file_name).replace(".raw", ""), patch_num), patch_rgb)
      del patch_rgb

  # t = 10
  # img = np.concatenate([np.arange(1, t * t * 1 + 1).reshape((t, t, 1)),
  #                       np.arange(t * t * 1 + 1, t * t * 2 + 1).reshape((t, t, 1)),
  #                       np.arange(t * t * 2 + 1, t * t * 3 + 1).reshape((t, t, 1))], axis=-1)
  # M = numpy.matlib.repmat(np.array([0, 1]).reshape((1, 2)), t, t // 2)[..., None]
  # M = np.concatenate([M, M, M], axis=-1)
