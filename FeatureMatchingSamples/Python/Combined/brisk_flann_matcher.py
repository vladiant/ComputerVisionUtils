# https://github.com/methylDragon/opencv-python-reference/blob/master/02%20OpenCV%20Feature%20Detection%20and%20Description.md
# Source: https://docs.opencv.org/3.4.4/dc/dc3/tutorial_py_matcher.html

import numpy as np
import cv2 as cv

img1 = cv.imread("box.png")  # queryImage
img2 = cv.imread("box_in_scene.png")  # trainImage

# Initiate BRISK detector
akaze = cv.BRISK_create()

# find the keypoints and descriptors with BRISK
kp1, des1 = akaze.detectAndCompute(img1, None)
kp2, des2 = akaze.detectAndCompute(img2, None)

FLANN_INDEX_LSH = 6
index_params = dict(
    algorithm=FLANN_INDEX_LSH,
    table_number=6,  # 12
    key_size=12,  # 20
    multi_probe_level=1,
)  # 2

# Then set number of searches. Higher is better, but takes longer
search_params = dict(checks=100)

# Initialize matches
flann = cv.FlannBasedMatcher(index_params, search_params)

# Find matches
matches = flann.match(des1, des2)

# Flags:
# cv.DRAW_MATCHES_FLAGS_DEFAULT
# cv.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG
# cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
# cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
img3 = cv.drawMatches(
    img1,
    kp1,
    img2,
    kp2,
    matches,
    None,
    flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS,
)

# Draw matches
cv.namedWindow("BRISK BF Matcher", cv.WINDOW_NORMAL)
cv.imshow("BRISK BF Matcher", img3)

# Calculate homography
# Consider point filtering
obj = []
scene = []
for match in matches:
    obj.append(kp1[match.queryIdx].pt)
    scene.append(kp2[match.trainIdx].pt)

# Calculate homography: Inliers and outliers
# RANSAC, LMEDS, RHO
H, _ = cv.findHomography(np.array(obj), np.array(scene), cv.RANSAC)

if H is not None:
    # Frame of the object image
    obj_points = np.array(
        [
            [0, 0],
            [img1.shape[1], 0],
            [img1.shape[1], img1.shape[0]],
            [0, img1.shape[0]],
        ],
        dtype=np.float,
    )

    # Check the sanity of the transformation
    warped_points = cv.perspectiveTransform(np.array([obj_points]), H)

    warped_image = np.copy(img2)
    cv.drawContours(
        warped_image, np.array([warped_points]).astype(np.int32), 0, (0, 0, 255)
    )

    cv.namedWindow("Warped Object", cv.WINDOW_NORMAL)
    cv.imshow("Warped Object", warped_image)
else:
    print("Error calculating perspective transformation")

cv.waitKey(0)
