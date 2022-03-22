import numpy as np
import scipy.ndimage
from filters import guided_filter, box_filter, guided_filter_MLRI, gauss1d, gauss2d
import numpy.matlib
import cv2


def red_blue_interpolation_second(green, red, blue, mask, eps):

  # inverse mask
  mask = mask.astype(np.float32)
  imr = np.zeros_like(mask[:, :, 0])
  img = np.zeros_like(mask[:, :, 0])
  imb = np.zeros_like(mask[:, :, 0])
  imr[mask[:, :, 0] == 0] = 1
  img[mask[:, :, 1] == 0] = 1
  imb[mask[:, :, 2] == 0] = 1
  imask = mask.copy()
  imask[:, :, 0] = imr
  imask[:, :, 1] = img
  imask[:, :, 2] = imb

  # Iterpolate R and B at G pixels
  # Step (i): iterative directional interpolation
  # initial linear interpolation
  F1 = np.array([1, 0, 1], dtype=np.float32).reshape((1, 3)) / 2
  Guider1 = red + scipy.ndimage.correlate(red, F1, mode="nearest") * mask[:, :, 1]
  Guideg1 = green
  Guideb1 = blue + scipy.ndimage.correlate(blue, F1, mode="nearest") * mask[:, :, 1]
  F2 = F1.T
  Guider2 = red + scipy.ndimage.correlate(red, F2, mode="nearest") * mask[:, :, 1]
  Guideg2 = green
  Guideb2 = blue + scipy.ndimage.correlate(blue, F2, mode="nearest") * mask[:, :, 1]

  # initial guided filter window size for RI
  h = 2
  v = 2
  # initial guided filter window size for MLRI
  h2 = 2
  v2 = 0
  # maximum iteration number
  itnum = 2

  # initialization of iteration criteria
  RI_w2R1 = np.ones_like(mask[:, :, 0]) * 1e32
  RI_w2R2 = np.ones_like(mask[:, :, 0]) * 1e32
  MLRI_w2R1 = np.ones_like(mask[:, :, 0]) * 1e32
  MLRI_w2R2 = np.ones_like(mask[:, :, 0]) * 1e32
  RI_w2B1 = np.ones_like(mask[:, :, 0]) * 1e32
  RI_w2B2 = np.ones_like(mask[:, :, 0]) * 1e32
  MLRI_w2B1 = np.ones_like(mask[:, :, 0]) * 1e32
  MLRI_w2B2 = np.ones_like(mask[:, :, 0]) * 1e32

  # initial guide image for RI/MLRI
  RI_Guideg1 = Guideg1
  RI_Guider1 = Guider1
  RI_Guideb1 = Guideb1
  RI_Guideg2 = Guideg2
  RI_Guider2 = Guider2
  RI_Guideb2 = Guideb2
  MLRI_Guideg1 = Guideg1
  MLRI_Guider1 = Guider1
  MLRI_Guideb1 = Guideb1
  MLRI_Guideg2 = Guideg2
  MLRI_Guider2 = Guider2
  MLRI_Guideb2 = Guideb2

  # initialization of interpolated R and B values
  RI_R1 = Guider1
  RI_R2 = Guider2
  MLRI_R1 = Guider1
  MLRI_R2 = Guider2
  RI_B1 = Guideb1
  RI_B2 = Guideb2
  MLRI_B1 = Guideb1
  MLRI_B2 = Guideb2

  # Iterative horizontal and vertical interpolation
  for ittime in range(1, itnum + 1):
    # generate horizontal and vertical tentative estimate by RI
    M = np.ones_like(mask[:, :, 0])  # mask
    RI_tentativeR1 = guided_filter(RI_Guideg1, RI_Guider1, M, h, v, eps)
    RI_tentativeB1 = guided_filter(RI_Guideg1, RI_Guideb1, M, h, v, eps)
    RI_tentativeR2 = guided_filter(RI_Guideg2, RI_Guider2, M, v, h, eps)
    RI_tentativeB2 = guided_filter(RI_Guideg2, RI_Guideb2, M, v, h, eps)
    # generate horizontal tentative estimate by MLRI
    F1 = np.array([-1, 0, 2, 0, -1], dtype=np.float32).reshape((1, 5))
    difR = scipy.ndimage.correlate(MLRI_Guider1, F1, mode="nearest")
    difG = scipy.ndimage.correlate(MLRI_Guideg1, F1, mode="nearest")
    difB = scipy.ndimage.correlate(MLRI_Guideb1, F1, mode="nearest")
    MLRI_tentativeR1 = guided_filter_MLRI(MLRI_Guideg1, MLRI_Guider1, M, difG, difR, imask[:, :, 1], h2, v2, eps)
    MLRI_tentativeB1 = guided_filter_MLRI(MLRI_Guideg1, MLRI_Guideb1, M, difG, difB, imask[:, :, 1], h2, v2, eps)
    # generate vertical tentative estimate by MLRI
    F2 = F1.T
    difR = scipy.ndimage.correlate(MLRI_Guider2, F2, mode="nearest")
    difG = scipy.ndimage.correlate(MLRI_Guideg2, F2, mode="nearest")
    difB = scipy.ndimage.correlate(MLRI_Guideb2, F2, mode="nearest")
    MLRI_tentativeR2 = guided_filter_MLRI(MLRI_Guideg2, MLRI_Guider2, M, difG, difR, imask[:, :, 1], v2, h2, eps)
    MLRI_tentativeB2 = guided_filter_MLRI(MLRI_Guideg2, MLRI_Guideb2, M, difG, difB, imask[:, :, 1], v2, h2, eps)
    # calculate residuals of RI and MLRI
    RI_residualR1 = (red - RI_tentativeR1) * imask[:, :, 1]
    RI_residualB1 = (blue - RI_tentativeB1) * imask[:, :, 1]
    RI_residualR2 = (red - RI_tentativeR2) * imask[:, :, 1]
    RI_residualB2 = (blue - RI_tentativeB2) * imask[:, :, 1]
    MLRI_residualR1 = (red - MLRI_tentativeR1) * imask[:, :, 1]
    MLRI_residualB1 = (blue - MLRI_tentativeB1) * imask[:, :, 1]
    MLRI_residualR2 = (red - MLRI_tentativeR2) * imask[:, :, 1]
    MLRI_residualB2 = (blue - MLRI_tentativeB2) * imask[:, :, 1]
    # horizontal and vertical linear interpolation of residuals
    K1 = np.array([1, 0, 1], dtype=np.float32).reshape((1, 3)) / 2
    RI_residualR1 = scipy.ndimage.correlate(RI_residualR1, K1, mode="nearest")
    RI_residualB1 = scipy.ndimage.correlate(RI_residualB1, K1, mode="nearest")
    MLRI_residualR1 = scipy.ndimage.correlate(MLRI_residualR1, K1, mode="nearest")
    MLRI_residualB1 = scipy.ndimage.correlate(MLRI_residualB1, K1, mode="nearest")
    K2 = K1.T
    RI_residualR2 = scipy.ndimage.correlate(RI_residualR2, K2, mode="nearest")
    RI_residualB2 = scipy.ndimage.correlate(RI_residualB2, K2, mode="nearest")
    MLRI_residualR2 = scipy.ndimage.correlate(MLRI_residualR2, K2, mode="nearest")
    MLRI_residualB2 = scipy.ndimage.correlate(MLRI_residualB2, K2, mode="nearest")

    # add tentative estimate
    RI_R1 = (RI_tentativeR1 + RI_residualR1) * mask[:, :, 1]
    RI_B1 = (RI_tentativeB1 + RI_residualB1) * mask[:, :, 1]
    RI_R2 = (RI_tentativeR2 + RI_residualR2) * mask[:, :, 1]
    RI_B2 = (RI_tentativeB2 + RI_residualB2) * mask[:, :, 1]
    MLRI_R1 = (MLRI_tentativeR1 + MLRI_residualR1) * mask[:, :, 1]
    MLRI_B1 = (MLRI_tentativeB1 + MLRI_residualB1) * mask[:, :, 1]
    MLRI_R2 = (MLRI_tentativeR2 + MLRI_residualR2) * mask[:, :, 1]
    MLRI_B2 = (MLRI_tentativeB2 + MLRI_residualB2) * mask[:, :, 1]

    # Step(ii): adaptive selection of iteration at each pixel
    # calculate iteration criteria
    RI_criR1 = (RI_Guider1 - RI_tentativeR1) * M
    RI_criB1 = (RI_Guideb1 - RI_tentativeB1) * M
    RI_criR2 = (RI_Guider2 - RI_tentativeR2) * M
    RI_criB2 = (RI_Guideb2 - RI_tentativeB2) * M
    MLRI_criR1 = (MLRI_Guider1 - MLRI_tentativeR1) * M
    MLRI_criB1 = (MLRI_Guideb1 - MLRI_tentativeB1) * M
    MLRI_criR2 = (MLRI_Guider2 - MLRI_tentativeR2) * M
    MLRI_criB2 = (MLRI_Guideb2 - MLRI_tentativeB2) * M

    # calculate gradient of iteration criteria
    F1 = np.array([-1, 0, 1], dtype=np.float32).reshape((1, 3))
    RI_difcriR1 = np.abs(scipy.ndimage.correlate(RI_criR1, F1, mode="nearest"))
    RI_difcriB1 = np.abs(scipy.ndimage.correlate(RI_criB1, F1, mode="nearest"))
    MLRI_difcriR1 = np.abs(scipy.ndimage.correlate(MLRI_criR1, F1, mode="nearest"))
    MLRI_difcriB1 = np.abs(scipy.ndimage.correlate(MLRI_criB1, F1, mode="nearest"))
    F2 = F1.T
    RI_difcriR2 = np.abs(scipy.ndimage.correlate(RI_criR2, F2, mode="nearest"))
    RI_difcriB2 = np.abs(scipy.ndimage.correlate(RI_criB2, F2, mode="nearest"))
    MLRI_difcriR2 = np.abs(scipy.ndimage.correlate(MLRI_criR2, F2, mode="nearest"))
    MLRI_difcriB2 = np.abs(scipy.ndimage.correlate(MLRI_criB2, F2, mode="nearest"))

    # absolute value of iteration criteria
    RI_criR1 = np.abs(RI_criR1)
    RI_criB1 = np.abs(RI_criB1)
    RI_criR2 = np.abs(RI_criR2)
    RI_criB2 = np.abs(RI_criB2)
    MLRI_criR1 = np.abs(MLRI_criR1)
    MLRI_criB1 = np.abs(MLRI_criB1)
    MLRI_criR2 = np.abs(MLRI_criR2)
    MLRI_criB2 = np.abs(MLRI_criB2)

    # directional map of iteration criteria
    RI_criR1 = RI_criR1 + RI_criB1
    RI_criB1 = RI_criB1 + RI_criR1
    RI_criR2 = RI_criR2 + RI_criB2
    RI_criB2 = RI_criB2 + RI_criR2
    MLRI_criR1 = MLRI_criR1 + MLRI_criB1
    MLRI_criB1 = MLRI_criB1 + MLRI_criR1
    MLRI_criR2 = MLRI_criR2 + MLRI_criB2
    MLRI_criB2 = MLRI_criB2 + MLRI_criR2

    # directional gradient map of iteration criteri
    RI_difcriR1 = RI_difcriR1 + RI_difcriB1
    RI_difcriB1 = RI_difcriB1 + RI_difcriR1
    RI_difcriR2 = RI_difcriR2 + RI_difcriB2
    RI_difcriB2 = RI_difcriB2 + RI_difcriR2
    MLRI_difcriR1 = MLRI_difcriR1 + MLRI_difcriB1
    MLRI_difcriB1 = MLRI_difcriB1 + MLRI_difcriR1
    MLRI_difcriR2 = MLRI_difcriR2 + MLRI_difcriB2
    MLRI_difcriB2 = MLRI_difcriB2 + MLRI_difcriR2

    # smoothing of iteration criteria
    sigma = 2
    F1 = gauss2d(sigma, 5)
    RI_criR1 = scipy.ndimage.correlate(RI_criR1, F1, mode="nearest")
    MLRI_criR1 = scipy.ndimage.correlate(MLRI_criR1, F1, mode="nearest")
    RI_criB1 = scipy.ndimage.correlate(RI_criB1, F1, mode="nearest")
    MLRI_criB1 = scipy.ndimage.correlate(MLRI_criB1, F1, mode="nearest")
    RI_difcriR1 = scipy.ndimage.correlate(RI_difcriR1, F1, mode="nearest")
    MLRI_difcriR1 = scipy.ndimage.correlate(MLRI_difcriR1, F1, mode="nearest")
    RI_difcriB1 = scipy.ndimage.correlate(RI_difcriB1, F1, mode="nearest")
    MLRI_difcriB1 = scipy.ndimage.correlate(MLRI_difcriB1, F1, mode="nearest")
    F2 = gauss2d(sigma, 5)
    RI_criR2 = scipy.ndimage.correlate(RI_criR2, F2, mode="nearest")
    MLRI_criR2 = scipy.ndimage.correlate(MLRI_criR2, F2, mode="nearest")
    RI_criB2 = scipy.ndimage.correlate(RI_criB2, F2, mode="nearest")
    MLRI_criB2 = scipy.ndimage.correlate(MLRI_criB2, F2, mode="nearest")
    RI_difcriR2 = scipy.ndimage.correlate(RI_difcriR2, F2, mode="nearest")
    MLRI_difcriR2 = scipy.ndimage.correlate(MLRI_difcriR2, F2, mode="nearest")
    RI_difcriB2 = scipy.ndimage.correlate(RI_difcriB2, F2, mode="nearest")
    MLRI_difcriB2 = scipy.ndimage.correlate(MLRI_difcriB2, F2, mode="nearest")

    # calcualte iteration criteria
    RI_wR1 = RI_criR1**2 * RI_difcriR1
    RI_wR2 = RI_criR2**2 * RI_difcriR2
    MLRI_wR1 = MLRI_criR1**2 * MLRI_difcriR1
    MLRI_wR2 = MLRI_criR2**2 * MLRI_difcriR2
    RI_wB1 = RI_criB1**2 * RI_difcriB1
    RI_wB2 = RI_criB2**2 * RI_difcriB2
    MLRI_wB1 = MLRI_criB1**2 * MLRI_difcriB1
    MLRI_wB2 = MLRI_criB2**2 * MLRI_difcriB2

    # find smaller criteria pixels
    RI_piR1 = RI_wR1 < RI_w2R1
    RI_piR2 = RI_wR2 < RI_w2R2
    MLRI_piR1 = MLRI_wR1 < MLRI_w2R1
    MLRI_piR2 = MLRI_wR2 < MLRI_w2R2
    RI_piB1 = RI_wB1 < RI_w2B1
    RI_piB2 = RI_wB2 < RI_w2B2
    MLRI_piB1 = MLRI_wB1 < MLRI_w2B1
    MLRI_piB2 = MLRI_wB2 < MLRI_w2B2

    # guide updating
    RI_Guider1 = red + RI_R1
    RI_Guideb1 = blue + RI_B1
    RI_Guider2 = red + RI_R2
    RI_Guideb2 = blue + RI_B2
    MLRI_Guider1 = red + MLRI_R1
    MLRI_Guideb1 = blue + MLRI_B1
    MLRI_Guider2 = red + MLRI_R2
    MLRI_Guideb2 = blue + MLRI_B2

    # select smallest iteration criteria at each pixel
    RI_R1[RI_piR1] = RI_Guider1[RI_piR1]
    MLRI_R1[MLRI_piR1] = MLRI_Guider1[MLRI_piR1]
    RI_R2[RI_piR2] = RI_Guider2[RI_piR2]
    MLRI_R2[MLRI_piR2] = MLRI_Guider2[MLRI_piR2]
    RI_B1[RI_piB1] = RI_Guideb1[RI_piB1]
    MLRI_B1[MLRI_piB1] = MLRI_Guideb1[MLRI_piB1]
    RI_B2[RI_piB2] = RI_Guideb2[RI_piB2]
    MLRI_B2[MLRI_piB2] = MLRI_Guideb2[MLRI_piB2]

    # update minimum iteration criteria
    RI_w2R1[RI_piR1] = RI_wR1[RI_piR1]
    RI_w2R2[RI_piR2] = RI_wR2[RI_piR2]
    RI_w2B1[RI_piB1] = RI_wB1[RI_piB1]
    RI_w2B2[RI_piB2] = RI_wB2[RI_piB2]
    MLRI_w2R1[MLRI_piR1] = MLRI_wR1[MLRI_piR1]
    MLRI_w2R2[MLRI_piR2] = MLRI_wR2[MLRI_piR2]
    MLRI_w2B1[MLRI_piB1] = MLRI_wB1[MLRI_piB1]
    MLRI_w2B2[MLRI_piB2] = MLRI_wB2[MLRI_piB2]

    # guided filter window size update
    h = h + 1
    v = v + 1
    h2 = h2 + 1
    v2 = v2 + 1

  # Step(iii): adaptive combining
  # combining weight
  RI_w2R1 = 1 / (RI_w2R1 + 1e-10)
  RI_w2R2 = 1 / (RI_w2R2 + 1e-10)
  MLRI_w2R1 = 1 / (MLRI_w2R1 + 1e-10)
  MLRI_w2R2 = 1 / (MLRI_w2R2 + 1e-10)
  RI_w2B1 = 1 / (RI_w2B1 + 1e-10)
  RI_w2B2 = 1 / (RI_w2B2 + 1e-10)
  MLRI_w2B1 = 1 / (MLRI_w2B1 + 1e-10)
  MLRI_w2B2 = 1 / (MLRI_w2B2 + 1e-10)

  wR = RI_w2R1 + RI_w2R2 + MLRI_w2R1 + MLRI_w2R2
  wB = RI_w2B1 + RI_w2B2 + MLRI_w2B1 + MLRI_w2B2

  # combining
  red2 = (RI_w2R1 * RI_R1 + RI_w2R2 * RI_R2 + MLRI_w2R1 * MLRI_R1 + MLRI_w2R2 * MLRI_R2) / (wR + 1e-32)
  blue2 = (RI_w2B1 * RI_B1 + RI_w2B2 * RI_B2 + MLRI_w2B1 * MLRI_B1 + MLRI_w2B2 * MLRI_B2) / (wB + 1e-32)

  # output of the second step
  red = red + red2 * mask[:, :, 1]
  blue = blue + blue2 * mask[:, :, 1]

  red = red.clip(0, 1)
  blue = blue.clip(0, 1)
  return red, blue
