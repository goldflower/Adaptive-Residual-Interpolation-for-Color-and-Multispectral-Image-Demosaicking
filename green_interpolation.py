import numpy as np
import scipy.ndimage
from filters import guided_filter, box_filter, guided_filter_MLRI, gauss1d, gauss2d
import numpy.matlib
import cv2


def green_interpolation(mosaic, mask, eps):
  # inverse mask
  imask = (mask == 0)
  raw_q = np.sum(mosaic, axis=-1)
  maskGr = np.zeros_like(raw_q)
  maskGb = np.zeros_like(raw_q)
  maskGr[1::2, ::2] = 1
  maskGb[::2, 1::2] = 1
  # Mrh: mask of horizontal R and G line
  # Mbh: mask of horizontal B and G line
  # Mrv: mask of vertical R and G line
  # Mbv: mask of vertical B and G line
  Mrh = mask[:, :, 0] + maskGr
  Mbh = mask[:, :, 2] + maskGb
  Mrv = mask[:, :, 0] + maskGb
  Mbv = mask[:, :, 2] + maskGr

  # Step(i): iterative directional interpolation
  # initial linear interpolation (Eq.(5))
  Kh = np.array([1 / 2, 0, 1 / 2]).reshape((1, -1))
  Kv = Kh.T

  raw_h = scipy.ndimage.correlate(raw_q.astype(np.float32), Kh, mode='nearest')
  raw_v = scipy.ndimage.correlate(raw_q.astype(np.float32), Kv, mode='nearest')
  # horizontal direction
  Guidegrh = mosaic[:, :, 1] * maskGr + raw_h * mask[:, :, 0]
  Guidegbh = mosaic[:, :, 1] * maskGb + raw_h * mask[:, :, 2]
  Guiderh = mosaic[:, :, 0] + raw_h * maskGr
  Guidebh = mosaic[:, :, 2] + raw_h * maskGb
  # vertical direction
  Guidegrv = mosaic[:, :, 1] * maskGb + raw_v * mask[:, :, 0]
  Guidegbv = mosaic[:, :, 1] * maskGr + raw_v * mask[:, :, 2]
  Guiderv = mosaic[:, :, 0] + raw_v * maskGb
  Guidebv = mosaic[:, :, 2] + raw_v * maskGr

  # initial guided filter window size for RI
  h = 2  # default = 2
  v = 1  # default = 1
  # initial guided filter window size for MLRI
  h2 = 4  # default=4
  v2 = 0
  # maximum iteration number
  itnum = 11  # default=11
  # initialization of horizontal and vertical iteration criteria
  RI_w2h = np.ones_like(maskGr) * 1e32
  RI_w2v = np.ones_like(maskGr) * 1e32
  MLRI_w2h = np.ones_like(maskGr) * 1e32
  MLRI_w2v = np.ones_like(maskGr) * 1e32

  # initial guide image for RI
  RI_Guidegrh = Guidegrh
  RI_Guidegbh = Guidegbh
  RI_Guiderh = Guiderh
  RI_Guidebh = Guidebh
  RI_Guidegrv = Guidegrv
  RI_Guidegbv = Guidegbv
  RI_Guiderv = Guiderv
  RI_Guidebv = Guidebv
  # initial guide image for MLRI
  MLRI_Guidegrh = Guidegrh
  MLRI_Guidegbh = Guidegbh
  MLRI_Guiderh = Guiderh
  MLRI_Guidebh = Guidebh
  MLRI_Guidegrv = Guidegrv
  MLRI_Guidegbv = Guidegbv
  MLRI_Guiderv = Guiderv
  MLRI_Guidebv = Guidebv
  # initialization of interpolated G values
  RI_Gh = Guidegrh + Guidegbh
  RI_Gv = Guidegrv + Guidegbv
  MLRI_Gh = Guidegrh + Guidegbh
  MLRI_Gv = Guidegrv + Guidegbv
  # iterative horizontal and vertical interpolation
  for it in range(itnum):
    # generate horizontal and vertical tentative estimate by RI (Eq.(6))
    RI_tentativeGrh = guided_filter(RI_Guiderh, RI_Guidegrh, Mrh, h, v, eps)
    RI_tentativeGbh = guided_filter(RI_Guidebh, RI_Guidegbh, Mbh, h, v, eps)
    RI_tentativeRh = guided_filter(RI_Guidegrh, RI_Guiderh, Mrh, h, v, eps)
    RI_tentativeBh = guided_filter(RI_Guidegbh, RI_Guidebh, Mbh, h, v, eps)
    RI_tentativeGrv = guided_filter(RI_Guiderv, RI_Guidegrv, Mrv, v, h, eps)
    RI_tentativeGbv = guided_filter(RI_Guidebv, RI_Guidegbv, Mbv, v, h, eps)
    RI_tentativeRv = guided_filter(RI_Guidegrv, RI_Guiderv, Mrv, v, h, eps)
    RI_tentativeBv = guided_filter(RI_Guidegbv, RI_Guidebv, Mbv, v, h, eps)
    test_result = RI_tentativeGrh + RI_tentativeGbh  # + RI_tentativeRh# + RI_tentativeBh
    test_result2 = RI_tentativeGrv + RI_tentativeGbv + RI_tentativeRv + RI_tentativeBv
    # generate horizontal tentative estimate by MLRI
    Fh = np.array([-1, 0, 2, 0, -1]).reshape(1, 5)
    # Fh = np.array([-2,1,-2]).reshape((1,3))
    difR = scipy.ndimage.correlate(MLRI_Guiderh, Fh, mode='nearest')
    difGr = scipy.ndimage.correlate(MLRI_Guidegrh, Fh, mode='nearest')
    difB = scipy.ndimage.correlate(MLRI_Guidebh, Fh, mode='nearest')
    difGb = scipy.ndimage.correlate(MLRI_Guidegbh, Fh, mode='nearest')
    MLRI_tentativeRh = guided_filter_MLRI(MLRI_Guidegrh, MLRI_Guiderh, Mrh, difGr, difR, mask[:, :, 0], h2, v2, eps)

    MLRI_tentativeBh = guided_filter_MLRI(MLRI_Guidegbh, MLRI_Guidebh, Mbh, difGb, difB, mask[:, :, 2], h2, v2, eps)
    MLRI_tentativeGrh = guided_filter_MLRI(MLRI_Guiderh, MLRI_Guidegrh, Mrh, difR, difGr, maskGr, h2, v2, eps)
    MLRI_tentativeGbh = guided_filter_MLRI(MLRI_Guidebh, MLRI_Guidegbh, Mbh, difB, difGb, maskGb, h2, v2, eps)

    # generate vertical tentative estimate by MLRI
    Fv = Fh.T
    difR = scipy.ndimage.correlate(MLRI_Guiderv, Fv, mode='nearest')

    difGr = scipy.ndimage.correlate(MLRI_Guidegrv, Fv, mode='nearest')
    difB = scipy.ndimage.correlate(MLRI_Guidebv, Fv, mode='nearest')
    difGb = scipy.ndimage.correlate(MLRI_Guidegbv, Fv, mode='nearest')
    MLRI_tentativeRv = guided_filter_MLRI(MLRI_Guidegrv, MLRI_Guiderv, Mrv, difGr, difR, mask[:, :, 0], v2, h2, eps, True)
    MLRI_tentativeBv = guided_filter_MLRI(MLRI_Guidegbv, MLRI_Guidebv, Mbv, difGb, difB, mask[:, :, 2], v2, h2, eps)
    MLRI_tentativeGrv = guided_filter_MLRI(MLRI_Guiderv, MLRI_Guidegrv, Mrv, difR, difGr, maskGb, v2, h2, eps)
    MLRI_tentativeGbv = guided_filter_MLRI(MLRI_Guidebv, MLRI_Guidegbv, Mbv, difB, difGb, maskGr, v2, h2, eps)
    # calculate residuals of RI and MLRI (Eq.(9))
    RI_residualGrh = (mosaic[:, :, 1] - RI_tentativeGrh) * maskGr
    RI_residualGbh = (mosaic[:, :, 1] - RI_tentativeGbh) * maskGb
    RI_residualRh = (mosaic[:, :, 0] - RI_tentativeRh) * mask[:, :, 0]
    RI_residualBh = (mosaic[:, :, 2] - RI_tentativeBh) * mask[:, :, 2]
    RI_residualGrv = (mosaic[:, :, 1] - RI_tentativeGrv) * maskGb
    RI_residualGbv = (mosaic[:, :, 1] - RI_tentativeGbv) * maskGr
    RI_residualRv = (mosaic[:, :, 0] - RI_tentativeRv) * mask[:, :, 0]
    RI_residualBv = (mosaic[:, :, 2] - RI_tentativeBv) * mask[:, :, 2]
    MLRI_residualGrh = (mosaic[:, :, 1] - MLRI_tentativeGrh) * maskGr
    MLRI_residualGbh = (mosaic[:, :, 1] - MLRI_tentativeGbh) * maskGb
    MLRI_residualRh = (mosaic[:, :, 0] - MLRI_tentativeRh) * mask[:, :, 0]
    MLRI_residualBh = (mosaic[:, :, 2] - MLRI_tentativeBh) * mask[:, :, 2]
    MLRI_residualGrv = (mosaic[:, :, 1] - MLRI_tentativeGrv) * maskGb
    MLRI_residualGbv = (mosaic[:, :, 1] - MLRI_tentativeGbv) * maskGr
    MLRI_residualRv = (mosaic[:, :, 0] - MLRI_tentativeRv) * mask[:, :, 0]
    MLRI_residualBv = (mosaic[:, :, 2] - MLRI_tentativeBv) * mask[:, :, 2]

    # horizontal and vertical linear interpolation of residuals (Eq.(10))
    Kh = np.array([1 / 2, 1, 1 / 2]).reshape((1, 3))

    RI_residualGrh = scipy.ndimage.correlate(RI_residualGrh, Kh, mode='nearest')
    RI_residualGbh = scipy.ndimage.correlate(RI_residualGbh, Kh, mode='nearest')
    RI_residualRh = scipy.ndimage.correlate(RI_residualRh, Kh, mode='nearest')
    RI_residualBh = scipy.ndimage.correlate(RI_residualBh, Kh, mode='nearest')
    MLRI_residualGrh = scipy.ndimage.correlate(MLRI_residualGrh, Kh, mode='nearest')
    MLRI_residualGbh = scipy.ndimage.correlate(MLRI_residualGbh, Kh, mode='nearest')
    MLRI_residualRh = scipy.ndimage.correlate(MLRI_residualRh, Kh, mode='nearest')
    MLRI_residualBh = scipy.ndimage.correlate(MLRI_residualBh, Kh, mode='nearest')
    Kv = Kh.T
    RI_residualGrv = scipy.ndimage.correlate(RI_residualGrv, Kv, mode='nearest')
    RI_residualGbv = scipy.ndimage.correlate(RI_residualGbv, Kv, mode='nearest')
    RI_residualRv = scipy.ndimage.correlate(RI_residualRv, Kv, mode='nearest')
    RI_residualBv = scipy.ndimage.correlate(RI_residualBv, Kv, mode='nearest')
    MLRI_residualGrv = scipy.ndimage.correlate(MLRI_residualGrv, Kv, mode='nearest')
    MLRI_residualGbv = scipy.ndimage.correlate(MLRI_residualGbv, Kv, mode='nearest')
    MLRI_residualRv = scipy.ndimage.correlate(MLRI_residualRv, Kv, mode='nearest')
    MLRI_residualBv = scipy.ndimage.correlate(MLRI_residualBv, Kv, mode='nearest')
    # add tentative estimate (Eq.(11))
    RI_Grh = (RI_tentativeGrh + RI_residualGrh) * mask[:, :, 0]
    RI_Gbh = (RI_tentativeGbh + RI_residualGbh) * mask[:, :, 2]
    RI_Rh = (RI_tentativeRh + RI_residualRh) * maskGr
    RI_Bh = (RI_tentativeBh + RI_residualBh) * maskGb
    RI_Grv = (RI_tentativeGrv + RI_residualGrv) * mask[:, :, 0]
    RI_Gbv = (RI_tentativeGbv + RI_residualGbv) * mask[:, :, 2]
    RI_Rv = (RI_tentativeRv + RI_residualRv) * maskGb
    RI_Bv = (RI_tentativeBv + RI_residualBv) * maskGr
    MLRI_Grh = (MLRI_tentativeGrh + MLRI_residualGrh) * mask[:, :, 0]
    MLRI_Gbh = (MLRI_tentativeGbh + MLRI_residualGbh) * mask[:, :, 2]
    MLRI_Rh = (MLRI_tentativeRh + MLRI_residualRh) * maskGr
    MLRI_Bh = (MLRI_tentativeBh + MLRI_residualBh) * maskGb
    MLRI_Grv = (MLRI_tentativeGrv + MLRI_residualGrv) * mask[:, :, 0]
    MLRI_Gbv = (MLRI_tentativeGbv + MLRI_residualGbv) * mask[:, :, 2]
    MLRI_Rv = (MLRI_tentativeRv + MLRI_residualRv) * maskGr
    MLRI_Bv = (MLRI_tentativeBv + MLRI_residualBv) * maskGr
   # Step(ii): adaptive selection of iteration at each pixel
    # calculate iteration criteria (Eq.(12))
    RI_criGrh = (RI_Guidegrh - RI_tentativeGrh) * Mrh
    RI_criGbh = (RI_Guidegbh - RI_tentativeGbh) * Mbh
    RI_criRh = (RI_Guiderh - RI_tentativeRh) * Mrh
    RI_criBh = (RI_Guidebh - RI_tentativeBh) * Mbh
    RI_criGrv = (RI_Guidegrv - RI_tentativeGrv) * Mrv
    RI_criGbv = (RI_Guidegbv - RI_tentativeGbv) * Mbv
    RI_criRv = (RI_Guiderv - RI_tentativeRv) * Mrv
    RI_criBv = (RI_Guidebv - RI_tentativeBv) * Mbv
    MLRI_criGrh = (MLRI_Guidegrh - MLRI_tentativeGrh) * Mrh
    MLRI_criGbh = (MLRI_Guidegbh - MLRI_tentativeGbh) * Mbh
    MLRI_criRh = (MLRI_Guiderh - MLRI_tentativeRh) * Mrh
    MLRI_criBh = (MLRI_Guidebh - MLRI_tentativeBh) * Mbh
    MLRI_criGrv = (MLRI_Guidegrv - MLRI_tentativeGrv) * Mrv
    MLRI_criGbv = (MLRI_Guidegbv - MLRI_tentativeGbv) * Mbv
    MLRI_criRv = (MLRI_Guiderv - MLRI_tentativeRv) * Mrv
    MLRI_criBv = (MLRI_Guidebv - MLRI_tentativeBv) * Mbv
    # calculate gradient of iteration criteria
    Fh = np.array([-1, 0, 1]).reshape((1, 3))
    RI_difcriGrh = np.abs(scipy.ndimage.correlate(RI_criGrh, Fh, mode='nearest'))
    RI_difcriGbh = np.abs(scipy.ndimage.correlate(RI_criGbh, Fh, mode='nearest'))
    RI_difcriRh = np.abs(scipy.ndimage.correlate(RI_criRh, Fh, mode='nearest'))
    RI_difcriBh = np.abs(scipy.ndimage.correlate(RI_criBh, Fh, mode='nearest'))
    MLRI_difcriGrh = np.abs(scipy.ndimage.correlate(MLRI_criGrh, Fh, mode='nearest'))
    MLRI_difcriGbh = np.abs(scipy.ndimage.correlate(MLRI_criGbh, Fh, mode='nearest'))
    MLRI_difcriRh = np.abs(scipy.ndimage.correlate(MLRI_criRh, Fh, mode='nearest'))
    MLRI_difcriBh = np.abs(scipy.ndimage.correlate(MLRI_criBh, Fh, mode='nearest'))
    Fv = Fh.T
    RI_difcriGrv = np.abs(scipy.ndimage.correlate(RI_criGrv, Fv, mode='nearest'))
    RI_difcriGbv = np.abs(scipy.ndimage.correlate(RI_criGbv, Fv, mode='nearest'))
    RI_difcriRv = np.abs(scipy.ndimage.correlate(RI_criRv, Fv, mode='nearest'))
    RI_difcriBv = np.abs(scipy.ndimage.correlate(RI_criBv, Fv, mode='nearest'))
    MLRI_difcriGrv = np.abs(scipy.ndimage.correlate(MLRI_criGrv, Fv, mode='nearest'))
    MLRI_difcriGbv = np.abs(scipy.ndimage.correlate(MLRI_criGbv, Fv, mode='nearest'))
    MLRI_difcriRv = np.abs(scipy.ndimage.correlate(MLRI_criRv, Fv, mode='nearest'))
    MLRI_difcriBv = np.abs(scipy.ndimage.correlate(MLRI_criBv, Fv, mode='nearest'))

    # absolute value of iteration criteria
    RI_criGrh = np.abs(RI_criGrh)
    RI_criGbh = np.abs(RI_criGbh)
    RI_criRh = np.abs(RI_criRh)
    RI_criBh = np.abs(RI_criBh)
    RI_criGrv = np.abs(RI_criGrv)
    RI_criGbv = np.abs(RI_criGbv)
    RI_criRv = np.abs(RI_criRv)
    RI_criBv = np.abs(RI_criBv)
    MLRI_criGrh = np.abs(MLRI_criGrh)
    MLRI_criGbh = np.abs(MLRI_criGbh)
    MLRI_criRh = np.abs(MLRI_criRh)
    MLRI_criBh = np.abs(MLRI_criBh)
    MLRI_criGrv = np.abs(MLRI_criGrv)
    MLRI_criGbv = np.abs(MLRI_criGbv)
    MLRI_criRv = np.abs(MLRI_criRv)
    MLRI_criBv = np.abs(MLRI_criBv)

    # add Gr and R (Gb and B) criteria residuals
    RI_criGRh = (RI_criGrh + RI_criRh) * Mrh
    RI_criGBh = (RI_criGbh + RI_criBh) * Mbh
    RI_criGRv = (RI_criGrv + RI_criRv) * Mrv
    RI_criGBv = (RI_criGbv + RI_criBv) * Mbv
    MLRI_criGRh = (MLRI_criGrh + MLRI_criRh) * Mrh
    MLRI_criGBh = (MLRI_criGbh + MLRI_criBh) * Mbh
    MLRI_criGRv = (MLRI_criGrv + MLRI_criRv) * Mrv
    MLRI_criGBv = (MLRI_criGbv + MLRI_criBv) * Mbv

    # add Gr and R (Gb and B) gradient of criteria residuals
    RI_difcriGRh = (RI_difcriGrh + RI_difcriRh) * Mrh
    RI_difcriGBh = (RI_difcriGbh + RI_difcriBh) * Mbh
    RI_difcriGRv = (RI_difcriGrv + RI_difcriRv) * Mrv
    RI_difcriGBv = (RI_difcriGbv + RI_difcriBv) * Mbv
    MLRI_difcriGRh = (MLRI_difcriGrh + MLRI_difcriRh) * Mrh
    MLRI_difcriGBh = (MLRI_difcriGbh + MLRI_difcriBh) * Mbh
    MLRI_difcriGRv = (MLRI_difcriGrv + MLRI_difcriRv) * Mrv
    MLRI_difcriGBv = (MLRI_difcriGbv + MLRI_difcriBv) * Mbv

    # directional map of iteration criteria
    RI_crih = RI_criGRh + RI_criGBh
    RI_criv = RI_criGRv + RI_criGBv
    MLRI_crih = MLRI_criGRh + MLRI_criGBh
    MLRI_criv = MLRI_criGRv + MLRI_criGBv

    # directional gradient map of iteration criteria
    RI_difcrih = RI_difcriGRh + RI_difcriGBh
    RI_difcriv = RI_difcriGRv + RI_difcriGBv
    MLRI_difcrih = MLRI_difcriGRh + MLRI_difcriGBh
    MLRI_difcriv = MLRI_difcriGRv + MLRI_difcriGBv

    # smoothing of iteration criteria
    sigma = 2
    Fh = gauss2d(sigma, 5)
    RI_crih = scipy.ndimage.correlate(RI_crih, Fh, mode='nearest')
    MLRI_crih = scipy.ndimage.correlate(MLRI_crih, Fh, mode='nearest')
    RI_difcrih = scipy.ndimage.correlate(RI_difcrih, Fh, mode='nearest')
    MLRI_difcrih = scipy.ndimage.correlate(MLRI_difcrih, Fh, mode='nearest')

    Fv = gauss2d(sigma, 5)
    RI_criv = scipy.ndimage.correlate(RI_criv, Fv, mode='nearest')
    MLRI_criv = scipy.ndimage.correlate(MLRI_criv, Fv, mode='nearest')
    RI_difcriv = scipy.ndimage.correlate(RI_difcriv, Fv, mode='nearest')
    MLRI_difcriv = scipy.ndimage.correlate(MLRI_difcriv, Fv, mode='nearest')

    # calcualte iteration criteria (Eq.(13))
    RI_wh = RI_crih**2 * RI_difcrih
    RI_wv = RI_criv**2 * RI_difcriv
    MLRI_wh = MLRI_crih**2 * MLRI_difcrih
    MLRI_wv = MLRI_criv**2 * MLRI_difcriv

    # find smaller criteria pixels
    RI_pih = RI_wh < RI_w2h
    RI_piv = RI_wv < RI_w2v
    MLRI_pih = MLRI_wh < MLRI_w2h
    MLRI_piv = MLRI_wv < MLRI_w2v

    RI_pih = (RI_wh < RI_w2h)
    RI_piv = (RI_wv < RI_w2v)
    MLRI_pih = (MLRI_wh < MLRI_w2h)
    MLRI_piv = (MLRI_wv < MLRI_w2v)
    # guide updating
    RI_Guidegrh = mosaic[:, :, 1] * maskGr + RI_Grh
    RI_Guidegbh = mosaic[:, :, 1] * maskGb + RI_Gbh
    RI_Guidegh = RI_Guidegrh + RI_Guidegbh
    RI_Guiderh = mosaic[:, :, 0] + RI_Rh
    RI_Guidebh = mosaic[:, :, 2] + RI_Bh
    RI_Guidegrv = mosaic[:, :, 1] * maskGb + RI_Grv
    RI_Guidegbv = mosaic[:, :, 1] * maskGr + RI_Gbv
    RI_Guidegv = RI_Guidegrv + RI_Guidegbv
    RI_Guiderv = mosaic[:, :, 0] + RI_Rv
    RI_Guidebv = mosaic[:, :, 2] + RI_Bv
    MLRI_Guidegrh = mosaic[:, :, 1] * maskGr + MLRI_Grh
    MLRI_Guidegbh = mosaic[:, :, 1] * maskGb + MLRI_Gbh
    MLRI_Guidegh = MLRI_Guidegrh + MLRI_Guidegbh
    MLRI_Guiderh = mosaic[:, :, 0] + MLRI_Rh
    MLRI_Guidebh = mosaic[:, :, 2] + MLRI_Bh
    MLRI_Guidegrv = mosaic[:, :, 1] * maskGb + MLRI_Grv
    MLRI_Guidegbv = mosaic[:, :, 1] * maskGr + MLRI_Gbv
    MLRI_Guidegv = MLRI_Guidegrv + MLRI_Guidegbv
    MLRI_Guiderv = mosaic[:, :, 0] + MLRI_Rv
    MLRI_Guidebv = mosaic[:, :, 2] + MLRI_Bv

    # select smallest iteration criteria at each pixel (Eq.(14))
    RI_Gh[RI_pih] = RI_Guidegh[RI_pih]
    MLRI_Gh[MLRI_pih] = MLRI_Guidegh[MLRI_pih]
    RI_Gv[RI_piv] = RI_Guidegv[RI_piv]
    MLRI_Gv[MLRI_piv] = MLRI_Guidegv[MLRI_piv]

    # update minimum iteration criteria
    RI_w2h[RI_pih] = RI_wh[RI_pih]
    RI_w2h[RI_pih] = RI_wh[RI_pih]
    RI_w2v[RI_piv] = RI_wv[RI_piv]
    RI_w2v[RI_piv] = RI_wv[RI_piv]
    MLRI_w2h[MLRI_pih] = MLRI_wh[MLRI_pih]
    MLRI_w2h[MLRI_pih] = MLRI_wh[MLRI_pih]
    MLRI_w2v[MLRI_piv] = MLRI_wv[MLRI_piv]
    MLRI_w2v[MLRI_piv] = MLRI_wv[MLRI_piv]

    # guided filter window size update
    h = h + 1
    v = v + 1
    h2 = h2 + 1
    v2 = v2 + 1

  # Step(iii): adaptive combining
  # combining weight (Eq.(16))
  RI_w2h = 1. / (RI_w2h + 1e-10)
  RI_w2v = 1. / (RI_w2v + 1e-10)
  MLRI_w2h = 1. / (MLRI_w2h + 1e-10)
  MLRI_w2v = 1. / (MLRI_w2v + 1e-10)
  w = RI_w2h + RI_w2v + MLRI_w2h + MLRI_w2v

  # combining (Eq.(15))
  green = (RI_w2h * RI_Gh + RI_w2v * RI_Gv + MLRI_w2h * MLRI_Gh + MLRI_w2v * MLRI_Gv) / (w + 1e-32)

  # final output
  green = green * imask[:, :, 1] + mosaic[:, :, 1]
  green = green.clip(0, 1)
  return green
