# https://github.com/methylDragon/opencv-python-reference/blob/master/02%20OpenCV%20Feature%20Detection%20and%20Description.md
# Source: https://docs.opencv.org/3.4.4/dc/dc3/tutorial_py_matcher.html

import cv2 as cv

img1 = cv.imread('box.png')  # queryImage
img2 = cv.imread('box_in_scene.png')  # trainImage

# Initiate KAZE detector
kaze = cv.KAZE_create()

# find the keypoints and descriptors with KAZE
kp1, des1 = kaze.detectAndCompute(img1, None)
kp2, des2 = kaze.detectAndCompute(img2, None)

# create BFMatcher object
# For KAZE: cv.NORM_L1 or cv.NORM_L2
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

# Match descriptors.
matches = bf.match(des1, des2)

# Sort them in the order of their distance.
matches = sorted(matches, key=lambda x: x.distance)

# Flags:
# cv.DRAW_MATCHES_FLAGS_DEFAULT
# cv.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG
# cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
# cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
img3 = cv.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

# Draw matches
cv.namedWindow('KAZE BF Matcher', cv.WINDOW_NORMAL)
cv.imshow('KAZE BF Matcher', img3)

cv.waitKey(0)
