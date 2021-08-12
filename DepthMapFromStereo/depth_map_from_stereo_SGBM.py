import sys
import cv2 as cv
import numpy as np


def f(x):
    return


mode_types = {
    0: cv.StereoSGBM_MODE_SGBM,
    1: cv.StereoSGBM_MODE_HH,
    2: cv.StereoSGBM_MODE_SGBM_3WAY,
    3: cv.StereoSGBM_MODE_HH4,
}

if len(sys.argv) < 3:
    print(f"Program usage: {sys.argv[0]} image_filename_left image_filename_right")
    exit(1)

# Position of left and right image matters
img_left = cv.imread(sys.argv[1], cv.IMREAD_GRAYSCALE)
img_right = cv.imread(sys.argv[2], cv.IMREAD_GRAYSCALE)

cv.namedWindow("Disparity", cv.WINDOW_NORMAL)

cv.createTrackbar("numDisparities", "Disparity", 1, 5, f)
cv.createTrackbar("blockSize", "Disparity", 7, 20, f)
cv.createTrackbar("minDisparity", "Disparity", -10, 10, f)
cv.createTrackbar("P1", "Disparity", 0, 100, f)
cv.createTrackbar("P2", "Disparity", 0, 100, f)
cv.createTrackbar("disp12MaxDiff", "Disparity", 0, 255, f)
cv.createTrackbar("preFilterCap", "Disparity", 100, 255, f)
cv.createTrackbar("uniquenessRatio", "Disparity", 5, 35, f)
cv.createTrackbar("speckleWindowSize", "Disparity", 50, 200, f)
cv.createTrackbar("speckleRange", "Disparity", 1, 3, f)
cv.createTrackbar("Mode Type", "Disparity", 0, len(mode_types) - 1, f)

while True:
    current_disparities = cv.getTrackbarPos("numDisparities", "Disparity") * 16 + 16
    current_blocksize = 2 * cv.getTrackbarPos("blockSize", "Disparity") + 5
    current_min_disparity = cv.getTrackbarPos("minDisparity", "Disparity")
    current_p1 = cv.getTrackbarPos("P1", "Disparity")
    current_p2 = cv.getTrackbarPos("P2", "Disparity") + current_p1
    current_disp12 = cv.getTrackbarPos("disp12MaxDiff", "Disparity") - 1
    current_prefiltercap = cv.getTrackbarPos("preFilterCap", "Disparity")
    current_uniqueness = cv.getTrackbarPos("uniquenessRatio", "Disparity")
    current_specklesize = cv.getTrackbarPos("speckleWindowSize", "Disparity")
    current_specklerange = cv.getTrackbarPos("speckleRange", "Disparity")
    current_mode_type = cv.getTrackbarPos("Mode Type", "Disparity")

    stereo = cv.StereoSGBM_create(
        numDisparities=current_disparities,
        blockSize=current_blocksize,
        minDisparity=current_min_disparity,
        P1=current_p1,
        P2=current_p2,
        disp12MaxDiff=current_disp12,
        preFilterCap=current_prefiltercap,
        uniquenessRatio=current_uniqueness,
        speckleWindowSize=current_specklesize,
        speckleRange=current_specklerange,
        mode=mode_types[current_mode_type],
    )

    # Return type int16
    disparity = stereo.compute(img_left, img_right)

    min_disparity, max_disparity, _, _ = cv.minMaxLoc(disparity)
    print(f"Min disparity: {min_disparity}  Max disparity: {max_disparity}")

    viewable = np.zeros((disparity.shape), dtype=np.uint8)

    # Normalize to be viewed
    viewable = cv.normalize(disparity, viewable, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

    cv.imshow("Disparity", viewable)
    if cv.waitKey(10) & 0xFF == 27:
        break

cv.destroyAllWindows()
