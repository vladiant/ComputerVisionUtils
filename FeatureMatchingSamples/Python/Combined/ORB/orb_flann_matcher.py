# https://github.com/methylDragon/opencv-python-reference/blob/master/02%20OpenCV%20Feature%20Detection%20and%20Description.md
# Source: https://docs.opencv.org/3.4.4/dc/dc3/tutorial_py_matcher.html

import cv2 as cv

img1 = cv.imread('box.png')  # queryImage
img2 = cv.imread('box_in_scene.png')  # trainImage

# Initiate ORB detector
orb = cv.ORB_create()

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=6,  # 12
                    key_size=12,  # 20
                    multi_probe_level=1)  # 2

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
img3 = cv.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

# Draw matches
cv.namedWindow('ORB BF Matcher', cv.WINDOW_NORMAL)
cv.imshow('ORB BF Matcher', img3)

cv.waitKey(0)
