import numpy as np
import numpy.matlib
import scipy


def gauss1d(sigma, filter_length=5):
  gauss_filter = [np.exp((-x**2) / (2 * sigma**2)) for x in range(-(filter_length // 2), int(filter_length // 2) + 1)]
  # The formula used above has been given in the instruction.
  return np.array(gauss_filter).reshape(filter_length, 1) / np.sum(gauss_filter)


def gauss2d(sigma, filter_length=5):
  gauss_filter1d = gauss1d(sigma, filter_length)
  gauss_filter2d = gauss_filter1d.dot(gauss_filter1d.T)
  return gauss_filter2d


def box_filter(img_src, h, v, debug=False):
  H, W = img_src.shape
  img_dest = np.zeros_like(img_src)
  if v != 0:
    # cumulative sum over Y axis
    im_cum = np.cumsum(img_src, 0)
    # difference over Y axis
    img_dest[0:v + 1, :] = im_cum[0 + v:2 * v + 1, :]
    img_dest[v + 1:H - v, :] = im_cum[2 * v + 1: H, :] - im_cum[0:H - 2 * v - 1, :]
    img_dest[H - v:H, :] = numpy.matlib.repmat(im_cum[H - 1:H, :], v, 1) - im_cum[H - 2 * v - 1:H - v - 1, :]
  if h != 0:
    if v != 0:
      # cumulative sum over X axis
      im_cum = np.cumsum(img_dest, 1)
    else:
      # cumulative sum over X axis
      im_cum = np.cumsum(img_src, 1)

    # difference over Y axis
    img_dest[:, :h + 1] = im_cum[:, h:2 * h + 1]
    img_dest[:, h + 1:W - h] = im_cum[:, 2 * h + 1:W] - im_cum[:, :W - 2 * h - 1]
    img_dest[:, W - h:W] = numpy.matlib.repmat(im_cum[:, W - 1:W], 1, h) - im_cum[:, W - 2 * h - 1:W - h - 1]
  return img_dest


def guided_filter(I, p, M, h, v, eps):
  H, W = I.shape

  N = box_filter(M, h, v)
  N[N <= 0] = 1
  # the size of each local patch N=(2r+1)^2 except for boundary pixels.
  N2 = box_filter(np.ones((H, W)), h, v)
  mean_I = box_filter(I * M, h, v) / N
  mean_p = box_filter(p * M, h, v) / N
  mean_Ip = box_filter(I * p * M, h, v) / N
  cov_Ip = mean_Ip - mean_I * mean_p
  mean_II = box_filter(I * I * M, h, v) / N
  var_I = mean_II - mean_I * mean_I

  # linear coefficients
  a = cov_Ip / (var_I + eps)
  b = mean_p - a * mean_I
  # weighted average
  dif = box_filter(I * I * M, h, v) * a * a + b * b * N + box_filter(p * p * M, h, v) + \
      2 * a * b * box_filter(I * M, h, v) - 2 * b * box_filter(p * M, h, v) - 2 * a * box_filter(p * I * M, h, v)
  dif[dif < 0] = 0
  dif = dif / N

  dif = dif**0.5
  dif[dif < 1e-3] = 1e-3
  dif = 1 / dif
  wdif = box_filter(dif, h, v)
  mean_a = box_filter(a * dif, h, v) / (wdif + 1e-4)
  mean_b = box_filter(b * dif, h, v) / (wdif + 1e-4)

  # final output
  q = mean_a * I + mean_b
  return q


def guidedfilter_diagonal(I, p, M, h, v, eps):

  # diagonal window setting
  r = h + v
  F = np.ones((2 * r + 1, 2 * r + 1))
  w = 2 * r + 1
  for i in range(1, v + 1):
    for t in range(1, 2 * i):
      F[t - 1, 2 * i - t - 1] = 0
      F[w - t, w - 2 * i + t] = 0
  for i in range(1, h + 1):
    for t in range(1, 2 * i):
      F[t - 1, w - 2 * i + t] = 0
      F[w - t, 2 * i - t - 1] = 0
  F2 = np.zeros((2 * r + 1, 2 * r + 1))
  F2[::2, ::2] = 1
  F2[1::2, 1::2] = 1
  F = F * F2
  # print(F)
  # image size
  H, W = I.shape
  # the number of the sammpled pixels in each local patch
  N = scipy.ndimage.correlate(M, F, mode='nearest')
  N[N == 0] = 1
  # the size of each local patch N=(2r+1)^2 except for boundary pixels.
  N2 = scipy.ndimage.correlate(np.ones((H, W)), F, mode='nearest')

  mean_I = scipy.ndimage.correlate(I * M, F, mode='nearest') / N
  mean_p = scipy.ndimage.correlate(p * M, F, mode='nearest') / N
  mean_Ip = scipy.ndimage.correlate(I * p * M, F, mode='nearest') / N

  cov_Ip = mean_Ip - mean_I * mean_p
  mean_II = scipy.ndimage.correlate(I * I * M, F, mode='nearest') / N
  var_I = mean_II - mean_I * mean_I

  # linear coefficients
  a = cov_Ip / (var_I + eps)
  b = mean_p - a * mean_I

  # weighted average
  dif = scipy.ndimage.correlate(I * I * M, F, mode='nearest') * a * a + b * b * N + scipy.ndimage.correlate(p * p * M, F, mode='nearest') \
      + 2 * a * b * scipy.ndimage.correlate(I * M, F, mode='nearest') - 2 * b * scipy.ndimage.correlate(p * M, F, mode='nearest') \
      - 2 * a * scipy.ndimage.correlate(p * I * M, F, mode='nearest')
  dif[dif < 0] = 0
  dif = dif / N
  dif = dif**0.5
  dif[dif < 1e-3] = 1e-3
  dif = 1 / dif
  wdif = scipy.ndimage.correlate(dif, F, mode='nearest')
  mean_a = scipy.ndimage.correlate(a * dif, F, mode='nearest') / (wdif + 1e-4)
  mean_b = scipy.ndimage.correlate(b * dif, F, mode='nearest') / (wdif + 1e-4)

  # final output
  q = mean_a * I + mean_b
  return q


def guided_filter_MLRI(G, R, mask, I, p, M, h, v, eps, debug=False):
  H, W = I.shape
  N = box_filter(M, h, v)
  N[N == 0] = 1
  N2 = box_filter(np.ones((H, W)), h, v, debug)
  mean_Ip = box_filter(I * p * M, h, v) / N
  mean_II = box_filter(I * I * M, h, v) / N
  a = mean_Ip / (mean_II + eps)
  N3 = box_filter(mask, h, v)
  N3[N3 <= 0] = 1

  mean_G = box_filter(G * mask, h, v) / N3
  mean_R = box_filter(R * mask, h, v) / N3
  b = mean_R - a * mean_G
  if debug:
    pass
  dif = box_filter(G * G * mask, h, v) * a * a + b * b * N3 + box_filter(R * R * mask, h, v) + \
      2 * a * b * box_filter(G * mask, h, v) - 2 * b * box_filter(R * mask, h, v) - 2 * a * box_filter(R * G * mask, h, v)
  dif[dif < 0] = 0
  dif = dif / N3
  dif = dif ** 0.5
  dif[dif < 1e-3] = 1e-3
  dif = 1 / dif
  wdif = box_filter(dif, h, v)
  mean_a = box_filter(a * dif, h, v) / (wdif + 1e-4)
  mean_b = box_filter(b * dif, h, v) / (wdif + 1e-4)
  q = mean_a * G + mean_b
  return q


def guidedfilter_MLRI_diagonal(G, R, mask, I, p, M, h, v, eps):

  # diagonal window setting
  r = h + v
  F = np.ones((2 * r + 1, 2 * r + 1))
  w = 2 * r + 1
  for i in range(1, v + 1):
    for t in range(1, 2 * i):
      F[t - 1, 2 * i - t - 1] = 0
      F[w - t, w - 2 * i + t] = 0
  for i in range(1, h + 1):
    for t in range(1, 2 * i):
      F[t - 1, w - 2 * i + t] = 0
      F[w - t, 2 * i - t - 1] = 0
  F2 = np.zeros((2 * r + 1, 2 * r + 1))
  F2[::2, ::2] = 1
  F2[1::2, 1::2] = 1
  F = F * F2
  # image size
  H, W = np.shape(I)
  # the number of the sammpled pixels in each local patch
  N = scipy.ndimage.correlate(M, F, mode="nearest")
  N[N == 0] = 1
  # the size of each local patch N=(2r+1)^2 except for boundary pixels.
  N2 = scipy.ndimage.correlate(np.ones((H, W)), F, mode="nearest")

  mean_Ip = scipy.ndimage.correlate(I * p * M, F, mode="nearest") / N
  mean_II = scipy.ndimage.correlate(I * I * M, F, mode="nearest") / N

  # linear coefficients
  a = mean_Ip / (mean_II + eps)
  N3 = scipy.ndimage.correlate(mask, F, mode="nearest")
  N3[N3 == 0] = 1
  mean_G = scipy.ndimage.correlate(G * mask, F, mode="nearest") / N3
  mean_R = scipy.ndimage.correlate(R * mask, F, mode="nearest") / N3
  b = mean_R - a * mean_G

  # weighted average
  dif = scipy.ndimage.correlate(G * G * mask, F, mode="nearest") * a * a + b * b * N3 + scipy.ndimage.correlate(R * R * mask, F, mode="nearest") \
      + 2 * a * b * scipy.ndimage.correlate(G * mask, F, mode="nearest") - 2 * b * scipy.ndimage.correlate(R * mask, F, mode="nearest") \
      - 2 * a * scipy.ndimage.correlate(R * G * mask, F, mode="nearest")
  dif = dif / N3
  dif[dif < 0] = 0
  dif = dif**0.5
  dif[dif < 1e-3] = 1e-3
  dif = 1 / dif
  wdif = scipy.ndimage.correlate(dif, F, mode="nearest")
  mean_a = scipy.ndimage.correlate(a * dif, F, mode="nearest") / (wdif + 1e-4)
  mean_b = scipy.ndimage.correlate(b * dif, F, mode="nearest") / (wdif + 1e-4)

  # final output
  q = mean_a * G + mean_b
  return q
