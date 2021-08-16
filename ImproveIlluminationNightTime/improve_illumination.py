# https://learnopencv.com/improving-illumination-in-night-time-images/
# https://github.com/spmallick/learnopencv/tree/master/Improving-Illumination-in-Night-Time-Images/python

import cv2
import numpy as np

from itertools import combinations_with_replacement
from collections import defaultdict

import numpy as np
from numpy.linalg import inv

B, G, R = 0, 1, 2  # index for convenience


def boxfilter(I, r):
    """Fast box filter implementation.
    Parameters
    ----------
    I:  a single channel/gray image data normalized to [0.0, 1.0]
    r:  window radius
    Return
    -----------
    The filtered image data.
    """
    M, N = I.shape
    dest = np.zeros((M, N))
    # print(I)

    # cumulative sum over Y axis (tate-houkou no wa)
    sumY = np.cumsum(I, axis=0)
    # print('sumY:{}'.format(sumY))
    # difference over Y axis
    dest[: r + 1] = sumY[r : 2 * r + 1]  # top r+1 lines
    dest[r + 1 : M - r] = sumY[2 * r + 1 :] - sumY[: M - 2 * r - 1]
    # print(sumY[2*r + 1:]) # from 2*r+1 to end lines
    # print(sumY[:M - 2*r - 1]) # same lines of above, from start
    # tile replicate sumY[-1] and line them up to match the shape of (r, 1)
    dest[-r:] = (
        np.tile(sumY[-1], (r, 1)) - sumY[M - 2 * r - 1 : M - r - 1]
    )  # bottom r lines

    # cumulative sum over X axis
    sumX = np.cumsum(dest, axis=1)
    # print('sumX:{}'.format(sumX))
    # difference over X axis
    dest[:, : r + 1] = sumX[:, r : 2 * r + 1]  # left r+1 columns
    dest[:, r + 1 : N - r] = sumX[:, 2 * r + 1 :] - sumX[:, : N - 2 * r - 1]
    dest[:, -r:] = (
        np.tile(sumX[:, -1][:, None], (1, r)) - sumX[:, N - 2 * r - 1 : N - r - 1]
    )  # right r columns

    # print(dest)

    return dest


def guided_filter(I, p, r=15, eps=1e-3):
    """Refine a filter under the guidance of another (RGB) image.
    Parameters
    -----------
    I:   an M * N * 3 RGB image for guidance.
    p:   the M * N filter to be guided. transmission is used for this case.
    r:   the radius of the guidance
    eps: epsilon for the guided filter
    Return
    -----------
    The guided filter.
    """
    M, N = p.shape
    base = boxfilter(np.ones((M, N)), r)  # this is needed for regularization

    # each channel of I filtered with the mean filter. this is myu.
    means = [boxfilter(I[:, :, i], r) / base for i in range(3)]

    # p filtered with the mean filter
    mean_p = boxfilter(p, r) / base

    # filter I with p then filter it with the mean filter
    means_IP = [boxfilter(I[:, :, i] * p, r) / base for i in range(3)]

    # covariance of (I, p) in each local patch
    covIP = [means_IP[i] - means[i] * mean_p for i in range(3)]

    # variance of I in each local patch: the matrix Sigma in ECCV10 eq.14
    var = defaultdict(dict)
    for i, j in combinations_with_replacement(range(3), 2):
        var[i][j] = boxfilter(I[:, :, i] * I[:, :, j], r) / base - means[i] * means[j]

    a = np.zeros((M, N, 3))
    for y, x in np.ndindex(M, N):
        #         rr, rg, rb
        # Sigma = rg, gg, gb
        #         rb, gb, bb
        Sigma = np.array(
            [
                [var[B][B][y, x], var[B][G][y, x], var[B][R][y, x]],
                [var[B][G][y, x], var[G][G][y, x], var[G][R][y, x]],
                [var[B][R][y, x], var[G][R][y, x], var[R][R][y, x]],
            ]
        )
        cov = np.array([c[y, x] for c in covIP])
        a[y, x] = np.dot(cov, inv(Sigma + eps * np.eye(3)))  # eq 14

    # ECCV10 eq.15
    b = mean_p - a[:, :, R] * means[R] - a[:, :, G] * means[G] - a[:, :, B] * means[B]

    # ECCV10 eq.16
    q = (
        boxfilter(a[:, :, R], r) * I[:, :, R]
        + boxfilter(a[:, :, G], r) * I[:, :, G]
        + boxfilter(a[:, :, B], r) * I[:, :, B]
        + boxfilter(b, r)
    ) / base

    return q


def get_illumination_channel(I, w):
    M, N, _ = I.shape
    # Padding for channels
    padded = np.pad(
        I, ((int(w / 2), int(w / 2)), (int(w / 2), int(w / 2)), (0, 0)), "edge"
    )
    darkch = np.zeros((M, N))
    brightch = np.zeros((M, N))

    for i, j in np.ndindex(darkch.shape):
        darkch[i, j] = np.min(padded[i : i + w, j : j + w, :])  # Dark channel
        brightch[i, j] = np.max(padded[i : i + w, j : j + w, :])  # Bright channel

    return darkch, brightch


def get_atmosphere(I, brightch, p=0.1):
    M, N = brightch.shape
    flatI = I.reshape(M * N, 3)  # Reshaping image array
    flatbright = brightch.ravel()  # Flattening image array

    searchidx = (-flatbright).argsort()[: int(M * N * p)]  # Sorting and slicing
    A = np.mean(flatI.take(searchidx, axis=0), dtype=np.float64, axis=0)
    return A


def get_initial_transmission(A, brightch):
    A_c = np.max(A)
    init_t = (brightch - A_c) / (1.0 - A_c)  # Finding initial transmission map
    return (init_t - np.min(init_t)) / (
        np.max(init_t) - np.min(init_t)
    )  # Normalized initial transmission map


def get_corrected_transmission(I, A, darkch, brightch, init_t, alpha, omega, w):
    im = np.empty(I.shape, I.dtype)
    for ind in range(0, 3):
        im[:, :, ind] = (
            I[:, :, ind] / A[ind]
        )  # Divide pixel values by atmospheric light

    dark_c, _ = get_illumination_channel(im, w)  # Dark channel transmission map
    dark_t = 1 - omega * dark_c  # Corrected dark transmission map
    corrected_t = (
        init_t  # Initializing corrected transmission map with initial transmission map
    )
    diffch = brightch - darkch

    for i in range(diffch.shape[0]):
        for j in range(diffch.shape[1]):
            if diffch[i, j] < alpha:
                corrected_t[i, j] = dark_t[i, j] * init_t[i, j]

    return np.abs(corrected_t)


def get_final_image(I, A, refined_t, tmin):
    # Duplicating the channel of 2D refined map to 3 channels
    refined_t_broadcasted = np.broadcast_to(
        refined_t[:, :, None], (refined_t.shape[0], refined_t.shape[1], 3)
    )
    J = (I - A) / (
        np.where(refined_t_broadcasted < tmin, tmin, refined_t_broadcasted)
    )  # Finding result
    return (J - np.min(J)) / (np.max(J) - np.min(J))  # Normalized image


def reduce_init_t(init_t):
    init_t = (init_t * 255).astype(np.uint8)
    xp = [0, 32, 255]
    fp = [0, 32, 48]
    x = np.arange(256)  # Creating array [0 .. 255]
    table = np.interp(x, xp, fp).astype(
        "uint8t"
    )  # Interpreting fp according to xp in range of x
    init_t = cv2.LUT(init_t, table)  # Lookup table
    init_t = init_t.astype(np.float64) / 255  # Normalizing the transmission map
    return init_t


def dehaze(I, tmin=0.1, w=15, alpha=0.4, omega=0.75, p=0.1, eps=1e-3, reduce=False):
    m, n, _ = I.shape
    Idark, Ibright = get_illumination_channel(I, w)
    A = get_atmosphere(I, Ibright, p)

    init_t = get_initial_transmission(A, Ibright)
    if reduce:
        init_t = reduce_init_t(init_t)
    corrected_t = get_corrected_transmission(
        I, A, Idark, Ibright, init_t, alpha, omega, w
    )

    normI = (I - I.min()) / (I.max() - I.min())
    refined_t = guided_filter(normI, corrected_t, w, eps)
    J_refined = get_final_image(I, A, refined_t, tmin)

    enhanced = (J_refined * 255).astype(np.uint8)
    f_enhanced = cv2.detailEnhance(enhanced, sigma_s=10, sigma_r=0.15)
    f_enhanced = cv2.edgePreservingFilter(f_enhanced, flags=1, sigma_s=64, sigma_r=0.2)
    return f_enhanced


im = cv2.imread("dark.png")
orig = im.copy()

I = np.asfarray(im)
I = I[:, :, :3] / 255

f_enhanced = dehaze(I)

cv2.imshow("Original", orig)
cv2.imshow("F_enhanced", f_enhanced)
cv2.waitKey(0)
