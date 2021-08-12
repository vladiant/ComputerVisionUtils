import sys
import cv2 as cv
import numpy as np


def f(x):
    return


if len(sys.argv) < 3:
    print(f"Program usage: {sys.argv[0]} image_filename_left image_filename_right")
    exit(1)

# Position of left and right image matters
img_left = cv.imread(sys.argv[1], cv.IMREAD_GRAYSCALE)
img_right = cv.imread(sys.argv[2], cv.IMREAD_GRAYSCALE)

cv.namedWindow("Disparity", cv.WINDOW_NORMAL)

cv.createTrackbar("numDisparities", "Disparity", 1, 5, f)
cv.createTrackbar("blockSize", "Disparity", 7, 20, f)

while True:
    current_disparities = cv.getTrackbarPos("numDisparities", "Disparity") * 16
    current_blocksize = 2 * cv.getTrackbarPos("blockSize", "Disparity") + 5

    stereo = cv.StereoBM_create(
        numDisparities=current_disparities, blockSize=current_blocksize
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
